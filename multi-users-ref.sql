-- =====================================================================
-- multi-users-ref.sql
-- Supabase: 멀티유저 / 멀티세션 / 벡터 검색을 위한 DB 스키마
-- ⚠ 실행 시 public.app_users / sessions / messages / documents 의
--    기존 데이터가 모두 삭제되고 스키마가 처음부터 다시 만들어집니다.
--
-- Supabase Dashboard → SQL Editor 에서 통째 실행.
-- 사용자별 분리는 앱 레벨 user_id 로 처리 (sessions → app_users 참조)
-- =====================================================================

-- 1) 확장 활성화 -------------------------------------------------------
create extension if not exists vector;

-- 2) 기존 객체 제거 (외래키 순서: 자식 테이블 → 세션 → 사용자) -----
drop function if exists public.match_documents(vector, double precision, integer, uuid) cascade;

drop table if exists public.messages cascade;
drop table if exists public.documents cascade;
drop table if exists public.sessions cascade;
drop table if exists public.app_users cascade;

drop function if exists public.set_updated_at() cascade;

-- 3) app_users 테이블 ---------------------------------------------------
create table public.app_users (
  id uuid primary key,
  login_id text not null unique,
  password_hash text not null,
  password_salt text not null,
  created_at timestamptz not null default now()
);

-- 4) sessions 테이블 ---------------------------------------------------
create table public.sessions (
  id uuid primary key,
  session_id uuid not null unique,
  user_id uuid not null references public.app_users(id) on delete cascade,
  title text not null default 'New Chat',
  created_at timestamptz not null default now(),
  updated_at timestamptz not null default now()
);

create or replace function public.set_updated_at()
returns trigger
language plpgsql
as $$
begin
  new.updated_at = now();
  return new;
end;
$$;

create trigger trg_sessions_updated_at
before update on public.sessions
for each row
execute function public.set_updated_at();

-- 5) messages 테이블 ---------------------------------------------------
create table public.messages (
  id bigserial primary key,
  session_id uuid not null references public.sessions(id) on delete cascade,
  user_id uuid not null references public.app_users(id) on delete cascade,
  role text not null check (role in ('user', 'ai', 'assistant')),
  content text not null,
  created_at timestamptz not null default now()
);

-- 6) documents 테이블 (pgvector) --------------------------------------
create table public.documents (
  id bigserial primary key,
  session_id uuid not null references public.sessions(id) on delete cascade,
  user_id uuid not null references public.app_users(id) on delete cascade,
  content_hash text not null,
  content text not null,
  metadata jsonb not null default '{}'::jsonb,
  embedding vector(1536) not null,
  created_at timestamptz not null default now(),
  unique (session_id, content_hash)
);

-- 7) 인덱스 -----------------------------------------------------------
create index if not exists idx_app_users_login_id on public.app_users (login_id);
create index if not exists idx_sessions_user_id on public.sessions (user_id);
create index if not exists idx_sessions_updated_at on public.sessions (updated_at desc);
create index if not exists idx_messages_session_created on public.messages (session_id, created_at);
create index if not exists idx_messages_user_id on public.messages (user_id);
create index if not exists idx_documents_session_id on public.documents (session_id);
create index if not exists idx_documents_user_id on public.documents (user_id);
create index if not exists idx_documents_metadata_session_id on public.documents ((metadata->>'session_id'));
create index if not exists idx_documents_embedding on public.documents
  using ivfflat (embedding vector_cosine_ops)
  with (lists = 100);

-- 8) RLS (Row Level Security) ----------------------------------------
alter table public.app_users enable row level security;
alter table public.sessions enable row level security;
alter table public.messages enable row level security;
alter table public.documents enable row level security;

drop policy if exists "app_users_open_all" on public.app_users;
drop policy if exists "sessions_open_all" on public.sessions;
drop policy if exists "messages_open_all" on public.messages;
drop policy if exists "documents_open_all" on public.documents;
drop policy if exists "sessions_select_own" on public.sessions;
drop policy if exists "sessions_insert_own" on public.sessions;
drop policy if exists "sessions_update_own" on public.sessions;
drop policy if exists "sessions_delete_own" on public.sessions;

drop policy if exists "messages_select_own" on public.messages;
drop policy if exists "messages_insert_own" on public.messages;
drop policy if exists "messages_update_own" on public.messages;
drop policy if exists "messages_delete_own" on public.messages;

drop policy if exists "documents_select_own" on public.documents;
drop policy if exists "documents_insert_own" on public.documents;
drop policy if exists "documents_update_own" on public.documents;
drop policy if exists "documents_delete_own" on public.documents;

create policy "app_users_open_all"
on public.app_users
for all
to anon, authenticated
using (true)
with check (true);

create policy "sessions_open_all"
on public.sessions
for all
to anon, authenticated
using (true)
with check (true);

create policy "messages_open_all"
on public.messages
for all
to anon, authenticated
using (true)
with check (true);

create policy "documents_open_all"
on public.documents
for all
to anon, authenticated
using (true)
with check (true);

-- 9) 유사도 검색 함수 (세션 필터링) -----------------------------------
create or replace function public.match_documents(
  query_embedding vector(1536),
  match_threshold double precision default 0.7,
  match_count integer default 10,
  filter_session_id uuid default null
)
returns table(
  id bigint,
  content text,
  metadata jsonb,
  similarity double precision
)
language sql
stable
security invoker
set search_path = public
as $$
  select
    d.id,
    d.content,
    d.metadata,
    1 - (d.embedding <=> query_embedding) as similarity
  from public.documents d
  where
    (filter_session_id is null or d.session_id = filter_session_id)
    and (1 - (d.embedding <=> query_embedding)) >= match_threshold
  order by d.embedding <=> query_embedding
  limit greatest(match_count, 1);
$$;

grant execute on function public.match_documents(vector, double precision, integer, uuid) to anon, authenticated;

-- PostgREST API 스키마 캐시 갱신(실패해도 무시 가능, 잠시 후 자동 반영됨)
notify pgrst, 'reload schema';
