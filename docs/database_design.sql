-- FocusPath / The Learning Hub database design for Supabase.
-- Run this in the Supabase SQL editor after enabling Supabase Auth.
-- Supabase Auth owns users/passwords; app data references auth.users(id).

create extension if not exists "pgcrypto";

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create table public.profiles (
  id uuid primary key references auth.users(id) on delete cascade,
  email text,
  full_name text,
  date_of_birth date,
  onboarding_completed_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.learner_profiles (
  user_id uuid primary key references auth.users(id) on delete cascade,
  preferred_methodology text not null default 'mix'
    check (preferred_methodology in ('text', 'video', 'audio', 'mix')),
  text_mix int not null default 40 check (text_mix between 0 and 100),
  video_mix int not null default 40 check (video_mix between 0 and 100),
  audio_mix int not null default 20 check (audio_mix between 0 and 100),
  focus_duration_minutes int not null default 25 check (focus_duration_minutes between 5 and 120),
  energy_level text not null default 'medium'
    check (energy_level in ('low', 'medium', 'high')),
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now(),
  constraint learner_profiles_mix_total check (text_mix + video_mix + audio_mix = 100)
);

create table public.user_xp (
  user_id uuid primary key references auth.users(id) on delete cascade,
  total_xp int not null default 0 check (total_xp >= 0),
  current_level int not null default 1 check (current_level >= 1),
  current_tier text not null default 'sprout'
    check (current_tier in ('sprout', 'builder', 'scholar', 'master')),
  completed_tracks int not null default 0 check (completed_tracks >= 0),
  total_focus_seconds int not null default 0 check (total_focus_seconds >= 0),
  updated_at timestamptz not null default now()
);

create table public.xp_tiers (
  tier text primary key,
  min_xp int not null unique check (min_xp >= 0),
  display_name text not null,
  position int not null unique
);

insert into public.xp_tiers (tier, min_xp, display_name, position)
values
  ('sprout', 0, 'Sprout', 1),
  ('builder', 500, 'Builder', 2),
  ('scholar', 1500, 'Scholar', 3),
  ('master', 3500, 'Master', 4);

create table public.learning_tracks (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  title text not null,
  description text,
  icon text not null default 'auto_stories',
  progress_percent numeric(5,2) not null default 0 check (progress_percent between 0 and 100),
  last_accessed_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.learning_modules (
  id uuid primary key default gen_random_uuid(),
  track_id uuid not null references public.learning_tracks(id) on delete cascade,
  title text not null,
  description text,
  position int not null default 0,
  summary_markdown text,
  video_url text,
  audio_url text,
  duration_seconds int,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.learning_concepts (
  id uuid primary key default gen_random_uuid(),
  module_id uuid not null references public.learning_modules(id) on delete cascade,
  title text not null,
  body_markdown text not null,
  image_url text,
  caption text,
  position int not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.source_materials (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  type text not null check (type in ('pdf', 'link', 'youtube')),
  title text,
  original_filename text,
  url text,
  storage_key text,
  mime_type text,
  file_size_bytes bigint,
  page_count int,
  extracted_text text,
  extracted_characters int,
  truncated boolean not null default false,
  processing_status text not null default 'pending'
    check (processing_status in ('pending', 'processing', 'ready', 'failed')),
  processing_error text,
  metadata jsonb not null default '{}',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.learning_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  track_id uuid references public.learning_tracks(id) on delete set null,
  topic text not null,
  learner_goal text,
  detail_level text not null default 'standard'
    check (detail_level in ('quick', 'standard', 'deep')),
  planned_duration_minutes int,
  actual_duration_seconds int check (actual_duration_seconds >= 0),
  xp_awarded int not null default 0 check (xp_awarded >= 0),
  xp_awarded_at timestamptz,
  status text not null default 'active'
    check (status in ('active', 'completed', 'abandoned')),
  started_at timestamptz not null default now(),
  ended_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.session_materials (
  session_id uuid not null references public.learning_sessions(id) on delete cascade,
  material_id uuid not null references public.source_materials(id) on delete cascade,
  position int not null default 0,
  created_at timestamptz not null default now(),
  primary key (session_id, material_id)
);

create table public.generated_notes (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references public.learning_sessions(id) on delete cascade,
  material_id uuid references public.source_materials(id) on delete set null,
  notes_markdown text not null,
  model text,
  prompt_version text not null default 'v1',
  created_at timestamptz not null default now()
);

create table public.chat_sessions (
  id uuid primary key default gen_random_uuid(),
  learning_session_id uuid not null references public.learning_sessions(id) on delete cascade,
  user_id uuid not null references auth.users(id) on delete cascade,
  title text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table public.chat_messages (
  id uuid primary key default gen_random_uuid(),
  chat_session_id uuid not null references public.chat_sessions(id) on delete cascade,
  role text not null check (role in ('user', 'assistant', 'system')),
  content_markdown text not null,
  metadata jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create table public.progress_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  session_id uuid references public.learning_sessions(id) on delete set null,
  track_id uuid references public.learning_tracks(id) on delete set null,
  module_id uuid references public.learning_modules(id) on delete set null,
  concept_id uuid references public.learning_concepts(id) on delete set null,
  event_type text not null,
  event_data jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create table public.xp_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references auth.users(id) on delete cascade,
  session_id uuid references public.learning_sessions(id) on delete set null,
  amount int not null check (amount > 0),
  reason text not null
    check (reason in ('session_completed', 'focus_time', 'quiz_completed', 'streak_bonus', 'manual_adjustment')),
  metadata jsonb not null default '{}',
  created_at timestamptz not null default now()
);

-- Optional future RAG table if you move embeddings from Pinecone to Supabase:
-- create extension if not exists vector;
-- create table public.document_chunks (
--   id uuid primary key default gen_random_uuid(),
--   material_id uuid not null references public.source_materials(id) on delete cascade,
--   chunk_index int not null,
--   page_start int,
--   page_end int,
--   content text not null,
--   embedding vector(768),
--   created_at timestamptz not null default now(),
--   unique (material_id, chunk_index)
-- );

create index idx_learning_tracks_user_id on public.learning_tracks(user_id);
create index idx_learning_tracks_user_last_accessed on public.learning_tracks(user_id, last_accessed_at desc nulls last);
create index idx_learning_modules_track_id on public.learning_modules(track_id);
create index idx_learning_concepts_module_id on public.learning_concepts(module_id);
create index idx_source_materials_user_id on public.source_materials(user_id);
create index idx_learning_sessions_user_id on public.learning_sessions(user_id);
create index idx_learning_sessions_track_id on public.learning_sessions(track_id);
create index idx_session_materials_material_id on public.session_materials(material_id);
create index idx_generated_notes_session_id on public.generated_notes(session_id);
create index idx_chat_sessions_learning_session_id on public.chat_sessions(learning_session_id);
create index idx_chat_sessions_user_id on public.chat_sessions(user_id);
create index idx_chat_messages_chat_session_id_created_at on public.chat_messages(chat_session_id, created_at);
create index idx_progress_events_user_id_created_at on public.progress_events(user_id, created_at);
create index idx_xp_events_user_id_created_at on public.xp_events(user_id, created_at);
create index idx_xp_events_session_id on public.xp_events(session_id);

create trigger profiles_set_updated_at
before update on public.profiles
for each row execute function public.set_updated_at();

create trigger learner_profiles_set_updated_at
before update on public.learner_profiles
for each row execute function public.set_updated_at();

create trigger user_xp_set_updated_at
before update on public.user_xp
for each row execute function public.set_updated_at();

create trigger learning_tracks_set_updated_at
before update on public.learning_tracks
for each row execute function public.set_updated_at();

create trigger learning_modules_set_updated_at
before update on public.learning_modules
for each row execute function public.set_updated_at();

create trigger learning_concepts_set_updated_at
before update on public.learning_concepts
for each row execute function public.set_updated_at();

create trigger source_materials_set_updated_at
before update on public.source_materials
for each row execute function public.set_updated_at();

create trigger learning_sessions_set_updated_at
before update on public.learning_sessions
for each row execute function public.set_updated_at();

create trigger chat_sessions_set_updated_at
before update on public.chat_sessions
for each row execute function public.set_updated_at();

create or replace function public.handle_new_user()
returns trigger
language plpgsql
security definer
set search_path = public
as $$
begin
  insert into public.profiles (id, email, full_name, date_of_birth)
  values (
    new.id,
    new.email,
    nullif(new.raw_user_meta_data ->> 'full_name', ''),
    nullif(new.raw_user_meta_data ->> 'date_of_birth', '')::date
  )
  on conflict (id) do update set
    email = excluded.email,
    full_name = coalesce(public.profiles.full_name, excluded.full_name),
    date_of_birth = coalesce(public.profiles.date_of_birth, excluded.date_of_birth);

  insert into public.learner_profiles (user_id)
  values (new.id)
  on conflict (user_id) do nothing;

  insert into public.user_xp (user_id)
  values (new.id)
  on conflict (user_id) do nothing;

  return new;
end;
$$;

create trigger on_auth_user_created
after insert on auth.users
for each row execute function public.handle_new_user();

alter table public.profiles enable row level security;
alter table public.learner_profiles enable row level security;
alter table public.user_xp enable row level security;
alter table public.learning_tracks enable row level security;
alter table public.learning_modules enable row level security;
alter table public.learning_concepts enable row level security;
alter table public.source_materials enable row level security;
alter table public.learning_sessions enable row level security;
alter table public.session_materials enable row level security;
alter table public.generated_notes enable row level security;
alter table public.chat_sessions enable row level security;
alter table public.chat_messages enable row level security;
alter table public.progress_events enable row level security;
alter table public.xp_tiers enable row level security;
alter table public.xp_events enable row level security;

create policy "Users can read own profile"
on public.profiles for select
using (auth.uid() = id);

create policy "Users can update own profile"
on public.profiles for update
using (auth.uid() = id)
with check (auth.uid() = id);

create policy "Users can insert own profile"
on public.profiles for insert
with check (auth.uid() = id);

create policy "Users can manage own learner profile"
on public.learner_profiles for all
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create policy "Users can read own xp summary"
on public.user_xp for select
using (auth.uid() = user_id);

create policy "Users can manage own tracks"
on public.learning_tracks for all
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create policy "Users can manage modules in own tracks"
on public.learning_modules for all
using (
  exists (
    select 1 from public.learning_tracks
    where learning_tracks.id = learning_modules.track_id
      and learning_tracks.user_id = auth.uid()
  )
)
with check (
  exists (
    select 1 from public.learning_tracks
    where learning_tracks.id = learning_modules.track_id
      and learning_tracks.user_id = auth.uid()
  )
);

create policy "Users can manage concepts in own modules"
on public.learning_concepts for all
using (
  exists (
    select 1
    from public.learning_modules
    join public.learning_tracks on learning_tracks.id = learning_modules.track_id
    where learning_modules.id = learning_concepts.module_id
      and learning_tracks.user_id = auth.uid()
  )
)
with check (
  exists (
    select 1
    from public.learning_modules
    join public.learning_tracks on learning_tracks.id = learning_modules.track_id
    where learning_modules.id = learning_concepts.module_id
      and learning_tracks.user_id = auth.uid()
  )
);

create policy "Users can manage own source materials"
on public.source_materials for all
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create policy "Users can manage own learning sessions"
on public.learning_sessions for all
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create policy "Users can manage own session materials"
on public.session_materials for all
using (
  exists (
    select 1 from public.learning_sessions
    where learning_sessions.id = session_materials.session_id
      and learning_sessions.user_id = auth.uid()
  )
)
with check (
  exists (
    select 1 from public.learning_sessions
    where learning_sessions.id = session_materials.session_id
      and learning_sessions.user_id = auth.uid()
  )
);

create policy "Users can manage notes in own sessions"
on public.generated_notes for all
using (
  exists (
    select 1 from public.learning_sessions
    where learning_sessions.id = generated_notes.session_id
      and learning_sessions.user_id = auth.uid()
  )
)
with check (
  exists (
    select 1 from public.learning_sessions
    where learning_sessions.id = generated_notes.session_id
      and learning_sessions.user_id = auth.uid()
  )
);

create policy "Users can manage own chat sessions"
on public.chat_sessions for all
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create policy "Users can manage messages in own chats"
on public.chat_messages for all
using (
  exists (
    select 1 from public.chat_sessions
    where chat_sessions.id = chat_messages.chat_session_id
      and chat_sessions.user_id = auth.uid()
  )
)
with check (
  exists (
    select 1 from public.chat_sessions
    where chat_sessions.id = chat_messages.chat_session_id
      and chat_sessions.user_id = auth.uid()
  )
);

create policy "Users can manage own progress events"
on public.progress_events for all
using (auth.uid() = user_id)
with check (auth.uid() = user_id);

create policy "Users can read xp tiers"
on public.xp_tiers for select
using (true);

create policy "Users can read own xp events"
on public.xp_events for select
using (auth.uid() = user_id);
