from dataclasses import dataclass


@dataclass(frozen=True)
class XpTier:
    key: str
    display_name: str
    min_xp: int


@dataclass(frozen=True)
class XpBreakdown:
    session_completion_xp: int
    focus_time_xp: int
    quiz_completion_xp: int
    milestone_bonus_xp: int

    @property
    def total(self) -> int:
        return (
            self.session_completion_xp
            + self.focus_time_xp
            + self.quiz_completion_xp
            + self.milestone_bonus_xp
        )


XP_TIERS = (
    XpTier(key="sprout", display_name="Sprout", min_xp=0),
    XpTier(key="builder", display_name="Builder", min_xp=500),
    XpTier(key="scholar", display_name="Scholar", min_xp=1500),
    XpTier(key="master", display_name="Master", min_xp=3500),
)

XP_PER_LEVEL = 250
BASE_SESSION_COMPLETION_XP = 50
XP_PER_FOCUS_MINUTE = 2
MAX_FOCUS_XP_MINUTES = 120
QUIZ_COMPLETION_XP = 25
FIVE_SESSION_MILESTONE_XP = 25


def calculate_level(total_xp: int) -> int:
    return max(1, (total_xp // XP_PER_LEVEL) + 1)


def calculate_tier(total_xp: int) -> XpTier:
    tier = XP_TIERS[0]
    for candidate in XP_TIERS:
        if total_xp >= candidate.min_xp:
            tier = candidate
    return tier


def calculate_next_level_xp(total_xp: int) -> int:
    return ((total_xp // XP_PER_LEVEL) + 1) * XP_PER_LEVEL


def calculate_session_xp(
    *,
    actual_duration_seconds: int,
    quiz_completed: bool,
    completed_track_count: int,
) -> XpBreakdown:
    focus_minutes = min(MAX_FOCUS_XP_MINUTES, max(0, actual_duration_seconds) // 60)
    milestone_bonus = FIVE_SESSION_MILESTONE_XP if completed_track_count > 0 and completed_track_count % 5 == 0 else 0

    return XpBreakdown(
        session_completion_xp=BASE_SESSION_COMPLETION_XP,
        focus_time_xp=focus_minutes * XP_PER_FOCUS_MINUTE,
        quiz_completion_xp=QUIZ_COMPLETION_XP if quiz_completed else 0,
        milestone_bonus_xp=milestone_bonus,
    )
