from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import httpx

from app.core.config import Settings
from app.services.chat_store import SessionXpBreakdown, UserXpSummary
from app.services.xp import calculate_level, calculate_next_level_xp, calculate_tier


class SupabaseXpError(RuntimeError):
    pass


@dataclass(frozen=True)
class PersistedXpAward:
    xp_awarded: int
    xp_summary: UserXpSummary


class SupabaseXpStore:
    def __init__(self, *, settings: Settings) -> None:
        if not settings.supabase_url or not settings.supabase_service_role_key:
            raise SupabaseXpError("Supabase XP persistence is not configured.")

        self.base_url = settings.supabase_url.rstrip("/") + "/rest/v1"
        self.service_role_key = settings.supabase_service_role_key

    async def complete_session(
        self,
        *,
        user_id: str,
        session_id: str,
        source_title: str,
        actual_duration_seconds: int,
        xp_awarded: int,
        xp_breakdown: SessionXpBreakdown | None,
    ) -> PersistedXpAward:
        self._validate_uuid(session_id, "session_id")
        completed_at = datetime.now(UTC).isoformat()

        async with httpx.AsyncClient(timeout=10.0) as client:
            await self._ensure_user_xp(client=client, user_id=user_id)
            existing_session = await self._get_learning_session(client=client, session_id=session_id, user_id=user_id)
            if existing_session and int(existing_session.get("xp_awarded") or 0) > 0:
                summary = await self._get_user_xp(client=client, user_id=user_id)
                return PersistedXpAward(
                    xp_awarded=int(existing_session["xp_awarded"]),
                    xp_summary=summary,
                )

            await self._upsert_completed_session(
                client=client,
                user_id=user_id,
                session_id=session_id,
                source_title=source_title,
                actual_duration_seconds=actual_duration_seconds,
                xp_awarded=xp_awarded,
                completed_at=completed_at,
                has_existing_session=existing_session is not None,
            )
            await self._insert_xp_event(
                client=client,
                user_id=user_id,
                session_id=session_id,
                xp_awarded=xp_awarded,
                xp_breakdown=xp_breakdown,
            )

            current_summary = await self._get_user_xp(client=client, user_id=user_id)
            updated_summary = self._apply_award_to_summary(
                summary=current_summary,
                xp_awarded=xp_awarded,
                actual_duration_seconds=actual_duration_seconds,
            )
            await self._update_user_xp(client=client, user_id=user_id, summary=updated_summary)
            return PersistedXpAward(xp_awarded=xp_awarded, xp_summary=updated_summary)

    async def get_xp_summary(self, *, user_id: str) -> UserXpSummary:
        async with httpx.AsyncClient(timeout=10.0) as client:
            await self._ensure_user_xp(client=client, user_id=user_id)
            return await self._get_user_xp(client=client, user_id=user_id)

    async def _ensure_user_xp(self, *, client: httpx.AsyncClient, user_id: str) -> None:
        response = await client.post(
            self._url("user_xp"),
            headers=self._headers(prefer="resolution=ignore-duplicates"),
            params={"on_conflict": "user_id"},
            json={"user_id": user_id},
        )
        if response.status_code not in {200, 201, 204, 409}:
            self._raise_response_error(response, "Could not ensure user_xp row.")

    async def _get_learning_session(
        self,
        *,
        client: httpx.AsyncClient,
        session_id: str,
        user_id: str,
    ) -> dict | None:
        response = await client.get(
            self._url("learning_sessions"),
            headers=self._headers(),
            params={
                "id": f"eq.{session_id}",
                "user_id": f"eq.{user_id}",
                "select": "id,status,xp_awarded,actual_duration_seconds,xp_awarded_at",
            },
        )
        self._raise_response_error(response, "Could not fetch learning session.")
        rows = response.json()
        return rows[0] if rows else None

    async def _upsert_completed_session(
        self,
        *,
        client: httpx.AsyncClient,
        user_id: str,
        session_id: str,
        source_title: str,
        actual_duration_seconds: int,
        xp_awarded: int,
        completed_at: str,
        has_existing_session: bool,
    ) -> None:
        payload = {
            "status": "completed",
            "ended_at": completed_at,
            "actual_duration_seconds": actual_duration_seconds,
            "xp_awarded": xp_awarded,
            "xp_awarded_at": completed_at,
        }

        if has_existing_session:
            response = await client.patch(
                self._url("learning_sessions"),
                headers=self._headers(),
                params={"id": f"eq.{session_id}", "user_id": f"eq.{user_id}"},
                json=payload,
            )
            self._raise_response_error(response, "Could not update completed learning session.")
            return

        response = await client.post(
            self._url("learning_sessions"),
            headers=self._headers(),
            json={
                "id": session_id,
                "user_id": user_id,
                "topic": source_title or "Study session",
                **payload,
            },
        )
        self._raise_response_error(response, "Could not create completed learning session.")

    async def _insert_xp_event(
        self,
        *,
        client: httpx.AsyncClient,
        user_id: str,
        session_id: str,
        xp_awarded: int,
        xp_breakdown: SessionXpBreakdown | None,
    ) -> None:
        metadata = {}
        if xp_breakdown:
            metadata["breakdown"] = {
                "session_completion_xp": xp_breakdown.session_completion_xp,
                "focus_time_xp": xp_breakdown.focus_time_xp,
                "quiz_completion_xp": xp_breakdown.quiz_completion_xp,
                "milestone_bonus_xp": xp_breakdown.milestone_bonus_xp,
            }

        response = await client.post(
            self._url("xp_events"),
            headers=self._headers(),
            json={
                "user_id": user_id,
                "session_id": session_id,
                "amount": xp_awarded,
                "reason": "session_completed",
                "metadata": metadata,
            },
        )
        self._raise_response_error(response, "Could not insert XP event.")

    async def _get_user_xp(self, *, client: httpx.AsyncClient, user_id: str) -> UserXpSummary:
        response = await client.get(
            self._url("user_xp"),
            headers=self._headers(),
            params={
                "user_id": f"eq.{user_id}",
                "select": (
                    "total_xp,current_level,current_tier,"
                    "completed_tracks,total_focus_seconds"
                ),
            },
        )
        self._raise_response_error(response, "Could not fetch user XP summary.")
        rows = response.json()
        if not rows:
            return UserXpSummary()

        row = rows[0]
        tier = calculate_tier(int(row["total_xp"]))
        return UserXpSummary(
            total_xp=int(row["total_xp"]),
            current_level=int(row["current_level"]),
            current_tier=str(row["current_tier"]),
            current_tier_display_name=tier.display_name,
            next_level_xp=calculate_next_level_xp(int(row["total_xp"])),
            completed_tracks=int(row["completed_tracks"]),
            total_focus_seconds=int(row["total_focus_seconds"]),
        )

    async def _update_user_xp(self, *, client: httpx.AsyncClient, user_id: str, summary: UserXpSummary) -> None:
        response = await client.patch(
            self._url("user_xp"),
            headers=self._headers(),
            params={"user_id": f"eq.{user_id}"},
            json={
                "total_xp": summary.total_xp,
                "current_level": summary.current_level,
                "current_tier": summary.current_tier,
                "completed_tracks": summary.completed_tracks,
                "total_focus_seconds": summary.total_focus_seconds,
            },
        )
        self._raise_response_error(response, "Could not update user XP summary.")

    @staticmethod
    def _apply_award_to_summary(
        *,
        summary: UserXpSummary,
        xp_awarded: int,
        actual_duration_seconds: int,
    ) -> UserXpSummary:
        total_xp = summary.total_xp + xp_awarded
        tier = calculate_tier(total_xp)
        return UserXpSummary(
            total_xp=total_xp,
            current_level=calculate_level(total_xp),
            current_tier=tier.key,
            current_tier_display_name=tier.display_name,
            next_level_xp=calculate_next_level_xp(total_xp),
            completed_tracks=summary.completed_tracks + 1,
            total_focus_seconds=summary.total_focus_seconds + actual_duration_seconds,
        )

    def _headers(self, *, prefer: str | None = None) -> dict[str, str]:
        headers = {
            "apikey": self.service_role_key,
            "Authorization": f"Bearer {self.service_role_key}",
            "Content-Type": "application/json",
        }
        if prefer:
            headers["Prefer"] = prefer
        return headers

    def _url(self, table: str) -> str:
        return f"{self.base_url}/{table}"

    @staticmethod
    def _validate_uuid(value: str, label: str) -> None:
        try:
            UUID(value)
        except ValueError as exc:
            raise SupabaseXpError(f"{label} must be a UUID to persist XP in Supabase.") from exc

    @staticmethod
    def _raise_response_error(response: httpx.Response, message: str) -> None:
        if response.status_code < 400:
            return
        raise SupabaseXpError(f"{message} Supabase returned {response.status_code}: {response.text}")
