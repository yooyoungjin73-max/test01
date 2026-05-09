"""
Supabase Postgres에 public.app_users 테이블을 생성합니다.
.env 에 DATABASE_URL(또는 SUPABASE_DB_URL) 이 있어야 합니다.

Supabase Dashboard → 프로젝트 → Settings → Database
→ Connection string → URI(Session mode 또는 Transaction 예: 포트 6543)
→ 비밀번호를 넣은 전체 문자열을 복사해 .env 에 설정:

  DATABASE_URL=postgresql://postgres.[ref]:YOUR_PASSWORD@aws-0-xxxx.pooler...

실행:
  uv run python prompt/10.multi-users/setup_app_users_db.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from sqlalchemy import create_engine, text


def _load_dotenv_layers() -> None:
    # .../prompt/10.multi-users/setup_app_users_db.py → repo = ai4ceo
    repo = Path(__file__).resolve().parents[2]
    app_dir = Path(__file__).resolve().parent
    for p in (repo / ".env", app_dir / ".env"):
        if p.is_file():
            load_dotenv(p, override=True)


def main() -> int:
    _load_dotenv_layers()
    url = (os.getenv("DATABASE_URL") or os.getenv("SUPABASE_DB_URL") or "").strip()
    if not url:
        print(
            "DATABASE_URL 또는 SUPABASE_DB_URL 이 .env 에 없습니다.\n"
            "Supabase → Settings → Database → Connection string 에서 비밀번호 포함 URI 를 복사해 넣어주세요.",
            file=sys.stderr,
        )
        return 1

    statements = [
        """
create table if not exists public.app_users (
  id uuid primary key,
  login_id text not null unique,
  password_hash text not null,
  password_salt text not null,
  created_at timestamptz not null default now()
);
""",
        """
create index if not exists idx_app_users_login_id on public.app_users (login_id);
""",
        """
alter table public.app_users enable row level security;
""",
        """
drop policy if exists "app_users_open_all" on public.app_users;
""",
        """
create policy "app_users_open_all"
on public.app_users
for all
to anon, authenticated
using (true)
with check (true);
""",
        """
notify pgrst, 'reload schema';
""",
    ]

    engine = create_engine(url, isolation_level="AUTOCOMMIT")
    try:
        with engine.connect() as conn:
            for stmt in statements:
                conn.execute(text(stmt.strip()))
        print("OK: public.app_users 준비됨.")
        return 0
    except Exception as e:
        print(f"실패: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
