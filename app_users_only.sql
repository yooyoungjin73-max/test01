-- =====================================================================
-- app_users_only.sql
-- 회원가입 오류(PGRST205: app_users 없음) 해결용.
-- 다른 테이블을 DROP 하지 않고 public.app_users 만 생성합니다.
-- Supabase → SQL Editor 에서 전체 실행 후 5~10초 뒤 앱 새로고침.
-- =====================================================================

create table if not exists public.app_users (
  id uuid primary key,
  login_id text not null unique,
  password_hash text not null,
  password_salt text not null,
  created_at timestamptz not null default now()
);

create index if not exists idx_app_users_login_id on public.app_users (login_id);

alter table public.app_users enable row level security;

drop policy if exists "app_users_open_all" on public.app_users;
create policy "app_users_open_all"
on public.app_users
for all
to anon, authenticated
using (true)
with check (true);

-- PostgREST 스키마 캐시 갱신(프로젝트마다 허용될 수 있음; 실패해도 무시 가능)
notify pgrst, 'reload schema';
