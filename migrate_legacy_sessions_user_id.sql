-- =====================================================================
-- migrate_legacy_sessions_user_id.sql
-- 구버전 multi-session-ref.sql 로 만든 DB에 멀티유저용 user_id 컬럼 추가
--
-- ⚠ 새로 시작 가능하면 multi-users-ref.sql 전체 실행이 더 안전합니다.
-- ⚠ 회원가입으로 최소 한 명의 행이 public.app_users 에 있어야 합니다.
-- Supabase SQL Editor 에서 실행 후 새로고침.
-- =====================================================================

do $$
declare
  v_uid uuid;
begin
  select id into v_uid
    from public.app_users
    order by created_at asc
    limit 1;

  if v_uid is null then
    raise exception '먼저 앱 회원가입으로 app_users 에 계정 한 개 이상 만들고 다시 실행하세요.';
  end if;

  -- sessions
  if not exists (
    select 1 from information_schema.columns
    where table_schema = 'public' and table_name = 'sessions' and column_name = 'user_id'
  ) then
    alter table public.sessions
      add column user_id uuid references public.app_users(id) on delete cascade;
  end if;

  update public.sessions set user_id = v_uid where user_id is null;
  alter table public.sessions alter column user_id set not null;

  -- messages
  if exists (
    select 1 from information_schema.tables
    where table_schema = 'public' and table_name = 'messages'
  ) then
    if not exists (
      select 1 from information_schema.columns
      where table_schema = 'public' and table_name = 'messages' and column_name = 'user_id'
    ) then
      alter table public.messages
        add column user_id uuid references public.app_users(id) on delete cascade;
    end if;

    update public.messages m
    set user_id = s.user_id
    from public.sessions s
    where m.session_id = s.id and m.user_id is null;

    update public.messages set user_id = v_uid where user_id is null;

    alter table public.messages alter column user_id set not null;
  end if;

  -- documents
  if exists (
    select 1 from information_schema.tables
    where table_schema = 'public' and table_name = 'documents'
  ) then
    if not exists (
      select 1 from information_schema.columns
      where table_schema = 'public' and table_name = 'documents' and column_name = 'user_id'
    ) then
      alter table public.documents
        add column user_id uuid references public.app_users(id) on delete cascade;
    end if;

    update public.documents d
    set user_id = s.user_id
    from public.sessions s
    where d.session_id = s.id and d.user_id is null;

    update public.documents set user_id = v_uid where user_id is null;

    alter table public.documents alter column user_id set not null;
  end if;
end
$$;

notify pgrst, 'reload schema';
