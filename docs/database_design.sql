-- FocusPath / The Learning Hub database design
-- PostgreSQL DDL intended for visualization/import tools such as DrawSQL.

create extension if not exists "pgcrypto";

create table users (
  id uuid primary key default gen_random_uuid(),
  email text not null unique,
  password_hash text,
  full_name text not null,
  date_of_birth date,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table learner_profiles (
  user_id uuid primary key references users(id) on delete cascade,
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

create table learning_tracks (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references users(id) on delete cascade,
  title text not null,
  description text,
  icon text,
  progress_percent numeric(5,2) not null default 0 check (progress_percent between 0 and 100),
  last_accessed_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table learning_modules (
  id uuid primary key default gen_random_uuid(),
  track_id uuid not null references learning_tracks(id) on delete cascade,
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

create table learning_concepts (
  id uuid primary key default gen_random_uuid(),
  module_id uuid not null references learning_modules(id) on delete cascade,
  title text not null,
  body_markdown text not null,
  image_url text,
  caption text,
  position int not null default 0,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table source_materials (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references users(id) on delete cascade,
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

create table learning_sessions (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references users(id) on delete cascade,
  track_id uuid references learning_tracks(id) on delete set null,
  topic text not null,
  learner_goal text,
  detail_level text not null default 'standard'
    check (detail_level in ('quick', 'standard', 'deep')),
  planned_duration_minutes int,
  status text not null default 'active'
    check (status in ('active', 'completed', 'abandoned')),
  started_at timestamptz not null default now(),
  ended_at timestamptz,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table session_materials (
  session_id uuid not null references learning_sessions(id) on delete cascade,
  material_id uuid not null references source_materials(id) on delete cascade,
  position int not null default 0,
  created_at timestamptz not null default now(),
  primary key (session_id, material_id)
);

create table generated_notes (
  id uuid primary key default gen_random_uuid(),
  session_id uuid not null references learning_sessions(id) on delete cascade,
  material_id uuid references source_materials(id) on delete set null,
  notes_markdown text not null,
  model text,
  prompt_version text not null default 'v1',
  created_at timestamptz not null default now()
);

create table chat_sessions (
  id uuid primary key default gen_random_uuid(),
  learning_session_id uuid not null references learning_sessions(id) on delete cascade,
  user_id uuid not null references users(id) on delete cascade,
  title text,
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create table chat_messages (
  id uuid primary key default gen_random_uuid(),
  chat_session_id uuid not null references chat_sessions(id) on delete cascade,
  role text not null check (role in ('user', 'assistant', 'system')),
  content_markdown text not null,
  metadata jsonb not null default '{}',
  created_at timestamptz not null default now()
);

create table progress_events (
  id uuid primary key default gen_random_uuid(),
  user_id uuid not null references users(id) on delete cascade,
  session_id uuid references learning_sessions(id) on delete set null,
  track_id uuid references learning_tracks(id) on delete set null,
  module_id uuid references learning_modules(id) on delete set null,
  concept_id uuid references learning_concepts(id) on delete set null,
  event_type text not null,
  event_data jsonb not null default '{}',
  created_at timestamptz not null default now()
);

-- Optional future RAG table. Enable pgvector first if you decide to use it:
create extension if not exists vector;
create table document_chunks (
  id uuid primary key default gen_random_uuid(),
  material_id uuid not null references source_materials(id) on delete cascade,
  chunk_index int not null,
  page_start int,
  page_end int,
  content text not null,
  embedding vector(768),
  created_at timestamptz not null default now(),
  unique (material_id, chunk_index)
);

create index idx_learner_profiles_user_id on learner_profiles(user_id);
create index idx_learning_tracks_user_id on learning_tracks(user_id);
create index idx_learning_modules_track_id on learning_modules(track_id);
create index idx_learning_concepts_module_id on learning_concepts(module_id);
create index idx_source_materials_user_id on source_materials(user_id);
create index idx_learning_sessions_user_id on learning_sessions(user_id);
create index idx_learning_sessions_track_id on learning_sessions(track_id);
create index idx_session_materials_material_id on session_materials(material_id);
create index idx_generated_notes_session_id on generated_notes(session_id);
create index idx_chat_sessions_learning_session_id on chat_sessions(learning_session_id);
create index idx_chat_sessions_user_id on chat_sessions(user_id);
create index idx_chat_messages_chat_session_id_created_at on chat_messages(chat_session_id, created_at);
create index idx_progress_events_user_id_created_at on progress_events(user_id, created_at);
