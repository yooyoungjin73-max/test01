"""
회원가입 별도 페이지
- user id / 비밀번호로 앱 내부 계정 생성
- 생성 즉시 메인 앱에서 로그인 가능
"""

import os
import hashlib
import secrets
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client


# pages/sign_up.py → 상위가 저장소(앱) 루트
REPO_ROOT = Path(__file__).resolve().parent.parent
APP_DIR = REPO_ROOT


def _load_dotenv_layers() -> None:
    """저장소 루트 .env (Streamlit Cloud 는 보통 Secrets 사용)."""
    for path in (REPO_ROOT / ".env",):
        if path.is_file():
            load_dotenv(path, override=True)


_load_dotenv_layers()


def _sync_secrets_to_env() -> None:
    if not hasattr(st, "secrets"):
        return
    try:
        secret_map = dict(st.secrets)
    except Exception:
        return
    for key in (
        "SUPABASE_URL",
        "SUPABASE_ANON_KEY",
        "SUPABASE_PUBLISHABLE_KEY",
        "SUPABASE_SERVICE_ROLE_KEY",
    ):
        if key in secret_map and not os.environ.get(key):
            os.environ[key] = str(secret_map[key])


_sync_secrets_to_env()


def _init_client():
    url = os.getenv("SUPABASE_URL")
    key = (
        os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_PUBLISHABLE_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception:
        return None


supabase: Client | None = _init_client()


def _signup_missing_table_error(exc: Exception) -> bool:
    t = str(exc).lower()
    return "pgrst205" in t or "could not find the table" in t


def _signup_bootstrap_sql() -> str:
    p = APP_DIR / "app_users_only.sql"
    if p.is_file():
        try:
            return p.read_text(encoding="utf-8").strip()
        except OSError:
            pass
    return (
        "create table if not exists public.app_users (\n"
        "  id uuid primary key,\n"
        "  login_id text not null unique,\n"
        "  password_hash text not null,\n"
        "  password_salt text not null,\n"
        "  created_at timestamptz not null default now()\n);\n\n"
        "create index if not exists idx_app_users_login_id "
        "on public.app_users (login_id);\n\n"
        "alter table public.app_users enable row level security;\n\n"
        'drop policy if exists "app_users_open_all" on public.app_users;\n'
        'create policy "app_users_open_all"\n'
        "on public.app_users\nfor all\nto anon, authenticated\n"
        "using (true)\nwith check (true);\n\n"
        "notify pgrst, 'reload schema';"
    )

st.set_page_config(page_title="회원가입", page_icon="🧾", layout="centered")

st.markdown(
    """
<style>
h1 { font-size: 1.4rem !important; font-weight: 600 !important; color: #ff69b4 !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; color: #ffd700 !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #1f77b4 !important; }
.stButton > button {
    background-color: #ff69b4 !important; color: white !important; border: none !important;
    border-radius: 5px !important; padding: 0.5rem 1rem !important; font-weight: bold !important;
}
.stButton > button:hover { background-color: #ff1493 !important; }
</style>
""",
    unsafe_allow_html=True,
)

st.title("📝 회원가입")
st.caption(
    "user id 와 비밀번호를 입력하면 즉시 계정이 생성됩니다. "
    "이후 메인 앱에서 같은 정보로 바로 로그인할 수 있습니다."
)

if not supabase:
    st.error(
        "Supabase 연결 정보가 없습니다. 관리자에게 `SUPABASE_URL`, "
        "`SUPABASE_ANON_KEY`(또는 `SUPABASE_PUBLISHABLE_KEY`) 설정을 요청해주세요."
    )
    st.stop()

try:
    supabase.table("app_users").select("id").limit(1).execute()
except Exception as probe_err:
    if _signup_missing_table_error(probe_err):
        st.error(
            "`public.app_users` 테이블이 없습니다. Supabase **SQL Editor**에서 아래를 실행한 후 "
            "10초 뒤 이 페이지를 새로고침하세요."
        )
        with st.expander("app_users 생성 SQL (복사)", expanded=True):
            st.code(_signup_bootstrap_sql(), language="sql")
        st.stop()
    st.error(f"Supabase 를 확인할 수 없습니다: {probe_err}")
    st.stop()

login_id = st.text_input("user id")
password = st.text_input("비밀번호 (8자 이상 권장)", type="password")
password2 = st.text_input("비밀번호 확인", type="password")

def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


col1, col2 = st.columns(2)
with col1:
    if st.button("가입하기", use_container_width=True):
        if not login_id or not password:
            st.warning("user id 와 비밀번호를 입력해주세요.")
        elif password != password2:
            st.warning("비밀번호가 일치하지 않습니다.")
        elif len(password) < 6:
            st.warning("비밀번호는 6자 이상이어야 합니다.")
        else:
            try:
                exists = (
                    supabase.table("app_users")
                    .select("id")
                    .eq("login_id", login_id.strip())
                    .limit(1)
                    .execute()
                )
                if exists.data:
                    st.error("이미 존재하는 user id 입니다.")
                else:
                    salt = secrets.token_hex(16)
                    password_hash = hash_password(password, salt)
                    insert_res = (
                        supabase.table("app_users")
                        .insert(
                            {
                                "id": str(uuid.uuid4()),
                                "login_id": login_id.strip(),
                                "password_hash": password_hash,
                                "password_salt": salt,
                            }
                        )
                        .execute()
                    )
                    if not insert_res.data:
                        st.error("회원가입에 실패했습니다. 다시 시도해주세요.")
                        st.stop()
                    st.success(
                        "✅ 회원가입이 완료되었습니다. 메인 앱에서 바로 로그인해주세요."
                    )
            except Exception as e:
                st.error(f"회원가입 실패: {str(e)}")

with col2:
    if st.button("메인으로 이동", use_container_width=True, type="secondary"):
        st.switch_page("multi-users-ref.py")
