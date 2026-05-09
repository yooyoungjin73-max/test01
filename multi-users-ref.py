"""
PDF 기반 멀티유저 RAG 챗봇
- Supabase Auth 미사용, 앱 내부 user_id/password 로그인
- 사용자별 세션/메시지/문서 분리 (앱 레벨 user_id 필터)
- 세션 저장/로드/삭제/화면초기화/제목보정/vectordb 버튼 제공
- Vector DB: Supabase pgvector + match_documents RPC
- LLM 모델: gpt-5.5 / claude-opus-4-7 / gemini-3-pro-preview (스트리밍)
- LLM API 키는 사이드바 입력이 기본이나, 로컬 임시 개발 시 .env 에서 채워 올 수 있음
- Supabase 키는 os.getenv 로 읽고 (레포/앱 폴더 .env 포함), Streamlit Secrets 는 비어 있는 키만 보완
"""

import os
import sys
import re
import uuid
import hashlib
import tempfile
import secrets
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional

import streamlit as st
from dotenv import load_dotenv
from supabase import create_client, Client

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_anthropic import ChatAnthropic
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks import CallbackManagerForRetrieverRun
from pydantic import Field, PrivateAttr

current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.insert(0, str(current_dir))


def _load_dotenv_layers() -> None:
    """로컬 임시 개발용: 레포 루트 → 앱 폴더 순으로 .env 로드 (나중 파일이 같은 키를 덮어씀)."""
    repo_root = current_dir.parent.parent
    for path in (repo_root / ".env", current_dir / ".env"):
        if path.is_file():
            load_dotenv(path, override=True)


_load_dotenv_layers()


# ---------------------------------------------------------------------------
# Streamlit Secrets → 환경 변수 동기화 (Streamlit Cloud 대응)
# ---------------------------------------------------------------------------
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


def _get_supabase_anon_key() -> Optional[str]:
    """anon key 우선, 새 명칭(publishable) 도 지원."""
    return (
        os.getenv("SUPABASE_ANON_KEY")
        or os.getenv("SUPABASE_PUBLISHABLE_KEY")
        or os.getenv("SUPABASE_SERVICE_ROLE_KEY")
    )


# ---------------------------------------------------------------------------
# 페이지 설정 + 글로벌 스타일 (multi-session-ref.py / ref.py 와 동일 톤)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="PDF 기반 멀티유저 RAG 챗봇",
    page_icon="📚",
    layout="wide",
)

st.markdown(
    """
<style>
h1 { font-size: 1.4rem !important; font-weight: 600 !important; color: #ff69b4 !important; }
h2 { font-size: 1.2rem !important; font-weight: 600 !important; color: #ffd700 !important; }
h3 { font-size: 1.1rem !important; font-weight: 600 !important; color: #1f77b4 !important; }
h4 { font-size: 1.1rem !important; font-weight: 600 !important; }
h5 { font-size: 1rem  !important; font-weight: 600 !important; }
h6 { font-size: 0.95rem !important; font-weight: 600 !important; }

.stChatMessage { font-size: 0.95rem !important; line-height: 1.5 !important; }
.stChatMessage p { font-size: 0.95rem !important; line-height: 1.5 !important; margin: 0.5rem 0 !important; }
.stChatMessage ul, .stChatMessage ol { font-size: 0.95rem !important; line-height: 1.5 !important; margin: 0.5rem 0 !important; }
.stChatMessage li { font-size: 0.95rem !important; line-height: 1.5 !important; margin: 0.3rem 0 !important; }
.stChatMessage strong, .stChatMessage b { font-size: 0.95rem !important; font-weight: 600 !important; }
.stChatMessage blockquote {
    font-size: 0.95rem !important; line-height: 1.5 !important; margin: 0.5rem 0 !important;
    padding-left: 1rem !important; border-left: 3px solid #e0e0e0 !important;
}
.stChatMessage code {
    font-size: 0.9rem !important; background-color: #f5f5f5 !important;
    padding: 0.2rem 0.4rem !important; border-radius: 3px !important;
}
.stChatMessage * { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important; }

.stButton > button {
    background-color: #ff69b4 !important; color: white !important; border: none !important;
    border-radius: 5px !important; padding: 0.5rem 1rem !important; font-weight: bold !important;
}
.stButton > button:hover { background-color: #ff1493 !important; }

.stSidebar .stButton > button {
    font-size: 0.7rem !important; padding: 0.3rem 0.65rem !important;
}
</style>
""",
    unsafe_allow_html=True,
)

st.markdown(
    """
<div style="text-align: center; margin-top: -4rem; margin-bottom: 0.5rem;">
    <h1 style="font-size: 2.5rem; font-weight: bold; margin: 0;">
        <span style="color: #1f77b4;">PDF</span>
        <span style="color: #ffffff; font-size: 0.7em;">기반</span>
        <span style="color: #9b59b6;">멀티유저</span>
        <span style="color: #ffd700;">RAG</span>
        <span style="color: #d62728; font-size: 0.7em;">챗봇</span>
    </h1>
</div>
""",
    unsafe_allow_html=True,
)
st.markdown("PDF 파일을 업로드하고 내용에 관해 질문해보세요!")


# ---------------------------------------------------------------------------
# 유틸
# ---------------------------------------------------------------------------
def sanitize_text(text: Optional[str]) -> str:
    if text is None:
        return ""
    cleaned = text.replace("\x00", "")
    cleaned = re.sub(r"[\x01-\x08\x0b\x0c\x0e-\x1f\x7f]", "", cleaned)
    return cleaned


def is_missing_table_error(error: Exception) -> bool:
    try:
        text = str(error).lower()
        return "pgrst205" in text or "could not find the table" in text
    except Exception:
        return False


def is_sessions_user_column_missing_error(error: Exception) -> bool:
    """예전 스키마: sessions 에 user_id 컬럼이 없음(42703)."""
    try:
        raw = str(error)
        t = raw.lower()
        if "user_id" not in t:
            return False
        if "42703" not in raw and "does not exist" not in t:
            return False
        return "sessions.user_id" in t.replace(" ", "") or (
            "sessions" in t and "user_id" in t
        )
    except Exception:
        return False


def _migrate_legacy_user_id_sql_text() -> str:
    p = current_dir / "migrate_legacy_sessions_user_id.sql"
    try:
        if p.is_file():
            return p.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return "-- migrate_legacy_sessions_user_id.sql 파일이 없습니다."


def render_stale_sessions_schema_help(where: str = "main") -> None:
    """구형 multi-session DB에 멀티유저용 user_id 컬럼 추가 안내."""
    sql = _migrate_legacy_user_id_sql_text()
    ctx = st if where == "main" else st.sidebar
    ctx.error(
        "**DB 스키마가 예전(multi-session)** 입니다. `sessions`(및 선택적으로 `messages`/`documents`)에 "
        "`user_id`가 없어 발생한 오류입니다. 아래 마이그레이션을 **SQL Editor**에서 실행하거나, "
        "데이터를 지워도 되면 **`multi-users-ref.sql` 전체**로 처음부터 만드세요."
    )
    with ctx.expander("user_id 추가 마이그레이션 SQL (복사)", expanded=True):
        st.code(sql, language="sql")
    ctx.caption("실행 순서: 회원가입으로 `app_users`에 계정 생성 → 위 SQL 실행 → 앱 새로고침.")


def _bootstrap_app_users_sql_text() -> str:
    p = current_dir / "app_users_only.sql"
    try:
        if p.is_file():
            return p.read_text(encoding="utf-8").strip()
    except OSError:
        pass
    return (
        """create table if not exists public.app_users (
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

notify pgrst, 'reload schema';"""
    )


def render_app_users_table_setup_help(where: str = "main") -> None:
    """PGRST205(app_users 미생성)일 때 실행할 SQL을 그대로 보여줌."""
    sql = _bootstrap_app_users_sql_text()
    ctx = st if where == "main" else st.sidebar
    ctx.error(
        "`public.app_users` 테이블이 Supabase에 없습니다. "
        "**Dashboard → SQL Editor**에서 아래를 전체 실행한 뒤 10초 정도 후 앱을 새로고침하세요."
    )
    with ctx.expander("app_users 생성 SQL (복사)", expanded=True):
        st.code(sql, language="sql")


def is_app_users_table_missing_in_rest() -> bool:
    """PostgREST가 app_users 미존재(PGRST205)로 거절하면 True."""
    if not supabase:
        return False
    try:
        supabase.table("app_users").select("id").limit(1).execute()
        return False
    except Exception as e:
        return is_missing_table_error(e)


# ---------------------------------------------------------------------------
# Supabase 클라이언트
# ---------------------------------------------------------------------------
@st.cache_resource
def init_supabase() -> Optional[Client]:
    url = os.getenv("SUPABASE_URL")
    key = _get_supabase_anon_key()
    if not url or not key:
        return None
    try:
        return create_client(url, key)
    except Exception as e:
        st.error(f"Supabase 연결 실패: {e}")
        return None


supabase = init_supabase()

if supabase and is_app_users_table_missing_in_rest():
    render_app_users_table_setup_help("main")


def current_user() -> Optional[Dict[str, Any]]:
    return st.session_state.get("app_user")


def current_user_id() -> Optional[str]:
    user = current_user()
    if not user:
        return None
    return user.get("id")


def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256(f"{salt}:{password}".encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# 세션 상태 초기화
# ---------------------------------------------------------------------------
def _init_session_state() -> None:
    defaults: Dict[str, Any] = {
        "conversation_memory": [],
        "retriever": None,
        "vectorstore": None,
        "processed_files": [],
        "chat_history": [],
        "current_session_id": str(uuid.uuid4()),
        "selected_model": "gpt-5.5",
        "sessions_loaded": False,
        "app_user": None,
        "openai_api_key": "",
        "anthropic_api_key": "",
        "gemini_api_key": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session_state()


def _hydrate_llm_keys_from_env() -> None:
    """사이드바가 비어 있을 때만 .env(이미 로드됨)의 키로 초기 채우기."""
    pairs = (
        ("openai_api_key", "OPENAI_API_KEY"),
        ("anthropic_api_key", "ANTHROPIC_API_KEY"),
        ("gemini_api_key", "GOOGLE_API_KEY"),
    )
    for sess_key, env_key in pairs:
        env_val = (os.getenv(env_key) or "").strip()
        if not env_val:
            continue
        cur = (st.session_state.get(sess_key) or "").strip()
        if not cur:
            st.session_state[sess_key] = env_val


_hydrate_llm_keys_from_env()


def apply_api_keys_to_env() -> None:
    """사이드바에서 입력받은 LLM API 키를 환경 변수로 주입."""
    if st.session_state.get("openai_api_key"):
        os.environ["OPENAI_API_KEY"] = st.session_state.openai_api_key.strip()
    if st.session_state.get("anthropic_api_key"):
        os.environ["ANTHROPIC_API_KEY"] = st.session_state.anthropic_api_key.strip()
    if st.session_state.get("gemini_api_key"):
        os.environ["GOOGLE_API_KEY"] = st.session_state.gemini_api_key.strip()


# ---------------------------------------------------------------------------
# 인증 (앱 내부 로그인 / 로그아웃)
# ---------------------------------------------------------------------------
def login_user(login_id: str, password: str) -> bool:
    if not supabase:
        st.error("Supabase 가 연결되어 있지 않습니다.")
        return False
    try:
        if not login_id.strip() or not password:
            st.error("user id 와 비밀번호를 입력해주세요.")
            return False
        user_res = (
            supabase.table("app_users")
            .select("id, login_id, password_hash, password_salt")
            .eq("login_id", login_id.strip())
            .limit(1)
            .execute()
        )
        if not user_res.data:
            st.error("존재하지 않는 user id 입니다.")
            return False
        user = user_res.data[0]
        expected_hash = hash_password(password, user["password_salt"])
        if expected_hash != user["password_hash"]:
            st.error("비밀번호가 올바르지 않습니다.")
            return False
        st.session_state.app_user = {"id": user["id"], "login_id": user["login_id"]}
        # 다른 유저로 전환 시 이전 컨텍스트 초기화
        st.session_state.chat_history = []
        st.session_state.conversation_memory = []
        st.session_state.processed_files = []
        st.session_state.retriever = None
        st.session_state.vectorstore = None
        st.session_state.current_session_id = str(uuid.uuid4())
        st.session_state.sessions_loaded = False
        return True
    except Exception as e:
        if is_missing_table_error(e):
            render_app_users_table_setup_help("sidebar")
        else:
            st.error(f"로그인 실패: {str(e)}")
        return False


def logout_user() -> None:
    for k in (
        "app_user",
        "chat_history",
        "conversation_memory",
        "processed_files",
        "retriever",
        "vectorstore",
    ):
        if k in st.session_state:
            st.session_state[k] = [] if isinstance(st.session_state[k], list) else None
    st.session_state.current_session_id = str(uuid.uuid4())
    st.session_state.sessions_loaded = False


@st.dialog("📝 회원가입")
def open_signup_dialog() -> None:
    """페이지 라우팅(pages/sign_up 없음) 실패 시에도 단일 파일 배포에서 가입 가능."""
    st.caption(
        "user id 와 비밀번호로 계정을 만듭니다. 가입 후 이 창을 닫고 같은 정보로 로그인하세요."
    )
    if not supabase:
        st.error("Supabase 가 연결되어 있지 않습니다.")
        return
    try:
        supabase.table("app_users").select("id").limit(1).execute()
    except Exception as probe_err:
        if is_missing_table_error(probe_err):
            st.error("`public.app_users` 테이블이 없습니다. SQL Editor 에서 아래를 실행한 뒤 새로고침하세요.")
            with st.expander("app_users 생성 SQL (복사)", expanded=True):
                st.code(_bootstrap_app_users_sql_text(), language="sql")
            return
        st.error(f"Supabase 확인 실패: {probe_err}")
        return

    su_login = st.text_input("user id", key="signup_dlg_login_id")
    su_pw = st.text_input("비밀번호 (6자 이상)", type="password", key="signup_dlg_pw")
    su_pw2 = st.text_input("비밀번호 확인", type="password", key="signup_dlg_pw2")
    if st.button("가입하기", type="primary", use_container_width=True, key="signup_dlg_submit"):
        if not su_login or not su_pw:
            st.warning("user id 와 비밀번호를 입력해주세요.")
        elif su_pw != su_pw2:
            st.warning("비밀번호가 일치하지 않습니다.")
        elif len(su_pw) < 6:
            st.warning("비밀번호는 6자 이상이어야 합니다.")
        else:
            try:
                exists = (
                    supabase.table("app_users")
                    .select("id")
                    .eq("login_id", su_login.strip())
                    .limit(1)
                    .execute()
                )
                if exists.data:
                    st.error("이미 존재하는 user id 입니다.")
                else:
                    salt = secrets.token_hex(16)
                    pw_hash = hash_password(su_pw, salt)
                    insert_res = (
                        supabase.table("app_users")
                        .insert(
                            {
                                "id": str(uuid.uuid4()),
                                "login_id": su_login.strip(),
                                "password_hash": pw_hash,
                                "password_salt": salt,
                            }
                        )
                        .execute()
                    )
                    if not insert_res.data:
                        st.error("회원가입에 실패했습니다. 다시 시도해주세요.")
                    else:
                        st.success("✅ 가입이 완료되었습니다. 이 창을 닫고 로그인하세요.")
            except Exception as e:
                if is_missing_table_error(e):
                    render_app_users_table_setup_help("main")
                else:
                    st.error(f"회원가입 실패: {e}")


# ---------------------------------------------------------------------------
# Retriever (사용자/세션 기반)
# ---------------------------------------------------------------------------
class SessionRetriever(BaseRetriever):
    """현재 사용자/세션 범위에서 유사 문서 검색."""

    k: int = Field(default=10, description="검색할 문서 수")

    _supabase: Client = PrivateAttr()
    _embeddings: OpenAIEmbeddings = PrivateAttr()
    _session_id: Optional[str] = PrivateAttr()

    def __init__(
        self,
        supabase_client: Client,
        embeddings: OpenAIEmbeddings,
        session_id: Optional[str] = None,
        k: int = 10,
    ):
        super().__init__(k=k)
        self._supabase = supabase_client
        self._embeddings = embeddings
        self._session_id = session_id

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        try:
            query_embedding = self._embeddings.embed_query(query)
            rpc_params = {
                "query_embedding": query_embedding,
                "match_threshold": 0.7,
                "match_count": self.k,
                "filter_session_id": self._session_id,
            }
            result = self._supabase.rpc("match_documents", rpc_params).execute()
            documents: List[Document] = []
            if result.data:
                for item in result.data:
                    metadata = item.get("metadata", {})
                    documents.append(
                        Document(
                            page_content=item.get("content", ""),
                            metadata=metadata if isinstance(metadata, dict) else {},
                        )
                    )
            return documents
        except Exception as e:
            st.error(f"Retriever 오류: {e}")
            return []


# ---------------------------------------------------------------------------
# 세션 / 메시지 / 문서 CRUD (전부 user_id 기반 + RLS)
# ---------------------------------------------------------------------------
def get_sessions() -> List[Dict[str, Any]]:
    if not supabase or not current_user_id():
        return []
    try:
        result = (
            supabase.table("sessions")
            .select("id, title, created_at, updated_at, session_id")
            .eq("user_id", current_user_id())
            .order("updated_at", desc=True)
            .limit(100)
            .execute()
        )
        return result.data or []
    except Exception as e:
        if is_missing_table_error(e):
            st.warning("Supabase 테이블이 아직 생성되지 않았습니다. `multi-users-ref.sql` 을 먼저 실행하세요.")
        elif is_sessions_user_column_missing_error(e):
            render_stale_sessions_schema_help("main")
        else:
            st.error(f"세션 목록 조회 실패: {e}")
        return []


def ensure_session_exists(session_id: str) -> bool:
    if not supabase or not current_user_id():
        return False
    try:
        existing = (
            supabase.table("sessions")
            .select("id")
            .eq("id", session_id)
            .eq("user_id", current_user_id())
            .limit(1)
            .execute()
        )
        if existing.data:
            return True
        updated_at_iso = datetime.now(timezone.utc).isoformat()
        created = supabase.table("sessions").insert(
            {
                "id": session_id,
                "session_id": session_id,
                "user_id": current_user_id(),
                "title": "New Chat",
                "updated_at": updated_at_iso,
            }
        ).execute()
        return bool(created.data)
    except Exception as e:
        if is_sessions_user_column_missing_error(e):
            render_stale_sessions_schema_help("sidebar")
        else:
            st.error(f"세션 초기 생성 실패: {e}")
        return False


def create_session() -> Optional[str]:
    if not supabase or not current_user_id():
        return None
    try:
        session_id = str(uuid.uuid4())
        updated_at_iso = datetime.now(timezone.utc).isoformat()
        result = supabase.table("sessions").insert(
            {
                "id": session_id,
                "session_id": session_id,
                "user_id": current_user_id(),
                "title": "New Chat",
                "updated_at": updated_at_iso,
            }
        ).execute()
        if result.data:
            return result.data[0]["id"]
        return None
    except Exception as e:
        if is_sessions_user_column_missing_error(e):
            render_stale_sessions_schema_help("sidebar")
        else:
            st.error(f"세션 생성 실패: {e}")
        return None


def generate_session_title(user_question: str, ai_response: str) -> str:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return user_question[:30] + ("..." if len(user_question) > 30 else "")
    try:
        llm = ChatOpenAI(model="gpt-5.5", temperature=0.7, openai_api_key=api_key)
        prompt = f"""다음 질문과 답변을 기반으로 핵심 키워드 2-3개를 추출하여 간결한 세션 제목을 생성해주세요.

사용자 질문: {user_question[:200]}
AI 답변: {ai_response[:300]}

요구사항:
- 핵심 키워드 2-3개를 조합하여 제목 생성
- 15-20자 이내의 간결한 제목
- 한글로 작성
- 따옴표나 특수문자 없이 작성
- 키워드만 조합한 형태 (예: "인공지능 활용 방안")
- 제목만 출력 (설명 없이)

제목:"""
        title = llm.invoke(prompt).content.strip().strip('"').strip("'").strip()
        if len(title) > 30:
            title = title[:27] + "..."
        return title or "New Chat"
    except Exception:
        return user_question[:30] + ("..." if len(user_question) > 30 else "")


def save_session(session_id: str) -> bool:
    if not supabase or not current_user_id():
        return False
    try:
        if not ensure_session_exists(session_id):
            return False

        updated_at_iso = datetime.now(timezone.utc).isoformat()
        title = "New Chat"

        # 첫 질문/답변 기반 제목 생성
        if len(st.session_state.chat_history) >= 2:
            user_msg = next((m["content"] for m in st.session_state.chat_history if m["role"] == "user"), "")
            ai_msg = next(
                (m["content"] for m in st.session_state.chat_history if m["role"] in ("assistant", "ai")),
                "",
            )
            if user_msg and ai_msg:
                try:
                    # 기존 제목이 New Chat 이거나 비어있을 때만 자동 생성
                    existing_session = (
                        supabase.table("sessions")
                        .select("title")
                        .eq("id", session_id)
                        .eq("user_id", current_user_id())
                        .limit(1)
                        .execute()
                    )
                    existing_title = (
                        existing_session.data[0].get("title") if existing_session.data else None
                    )
                    if not existing_title or existing_title == "New Chat":
                        title = generate_session_title(user_msg, ai_msg)
                    else:
                        title = existing_title
                except Exception:
                    title = generate_session_title(user_msg, ai_msg)

        try:
            supabase.table("sessions").update(
                {"title": title, "updated_at": updated_at_iso}
            ).eq("id", session_id).eq("user_id", current_user_id()).execute()
        except Exception as e:
            st.error(f"세션 업데이트 실패: {e}")
            return False

        # 기존 메시지 (중복 방지용)
        existing_pairs: List[tuple] = []
        try:
            messages_result = (
                supabase.table("messages")
                .select("role, content")
                .eq("session_id", session_id)
                .eq("user_id", current_user_id())
                .execute()
            )
            if messages_result.data:
                existing_pairs = [
                    (m.get("role"), (m.get("content", "") or "")[:1000])
                    for m in messages_result.data
                ]
        except Exception:
            existing_pairs = []

        saved = 0
        skipped = 0
        for msg in st.session_state.chat_history:
            try:
                role = msg.get("role")
                if not role:
                    continue
                if role == "assistant":
                    role = "ai"

                content = str(msg.get("content") or "")
                MAX_LEN = 1_000_000
                if len(content) > MAX_LEN:
                    content = content[:MAX_LEN] + "\n\n[메시지가 너무 길어 일부가 잘렸습니다...]"
                if not content.strip():
                    continue

                preview = content[:1000]
                if (role, preview) in existing_pairs:
                    skipped += 1
                    continue

                supabase.table("messages").insert(
                    {
                        "session_id": str(session_id),
                        "user_id": current_user_id(),
                        "role": str(role),
                        "content": content,
                    }
                ).execute()
                saved += 1
                existing_pairs.append((role, preview))
            except Exception as e:
                st.warning(f"메시지 저장 실패: {e}")
                continue

        if saved > 0:
            st.info(f"✅ {saved}개 메시지를 저장했습니다. (건너뜀: {skipped}개)")
        return True
    except Exception as e:
        st.error(f"세션 저장 중 오류: {e}")
        return False


def load_session(session_id: str) -> bool:
    if not supabase or not current_user_id():
        return False
    try:
        result = (
            supabase.table("messages")
            .select("id, role, content, created_at")
            .eq("session_id", session_id)
            .eq("user_id", current_user_id())
            .execute()
        )
        if result.data:
            result.data.sort(key=lambda x: x.get("created_at", ""))

        st.session_state.chat_history = []
        st.session_state.conversation_memory = []

        if result.data:
            for msg in result.data:
                role = msg.get("role", "")
                content = msg.get("content", "")
                if not role or not content:
                    continue
                display_role = "assistant" if role == "ai" else role
                st.session_state.chat_history.append({"role": display_role, "content": content})
                if role == "user":
                    st.session_state.conversation_memory.append(f"사용자: {content}")
                elif role == "ai":
                    st.session_state.conversation_memory.append(f"AI: {content}")

        # Retriever 복원
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            try:
                embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                st.session_state.retriever = SessionRetriever(
                    supabase, embeddings, session_id, k=10
                )
            except Exception as e:
                st.session_state.retriever = None
                st.warning(f"Retriever 복원 실패: {e}")
        else:
            st.session_state.retriever = None

        return True
    except Exception as e:
        if is_missing_table_error(e):
            st.warning("세션 관련 테이블이 없어 로드할 수 없습니다. `multi-users-ref.sql` 을 먼저 적용하세요.")
        else:
            st.error(f"세션 로드 실패: {e}")
        return False


def delete_session(session_id: str) -> bool:
    if not supabase or not current_user_id():
        return False
    try:
        try:
            supabase.table("documents").delete().eq("session_id", session_id).eq(
                "user_id", current_user_id()
            ).execute()
        except Exception as e:
            st.warning(f"문서 삭제 중 일부 오류: {e}")

        supabase.table("sessions").delete().eq("id", session_id).eq(
            "user_id", current_user_id()
        ).execute()
        return True
    except Exception as e:
        if is_missing_table_error(e):
            st.warning("세션 관련 테이블이 없어 삭제할 수 없습니다.")
        else:
            st.error(f"세션 삭제 실패: {e}")
        return False


def fix_session_title(session_id: str) -> bool:
    """현재 세션의 제목을 첫 질문/답변을 기반으로 다시 생성."""
    if not supabase or not current_user_id():
        return False
    try:
        if not st.session_state.chat_history:
            st.warning("제목 보정에 사용할 대화가 없습니다.")
            return False
        user_msg = next((m["content"] for m in st.session_state.chat_history if m["role"] == "user"), "")
        ai_msg = next(
            (m["content"] for m in st.session_state.chat_history if m["role"] in ("assistant", "ai")),
            "",
        )
        if not (user_msg and ai_msg):
            st.warning("질문/답변 쌍이 충분하지 않습니다.")
            return False
        new_title = generate_session_title(user_msg, ai_msg)
        updated_at_iso = datetime.now(timezone.utc).isoformat()
        supabase.table("sessions").update(
            {"title": new_title, "updated_at": updated_at_iso}
        ).eq("id", session_id).eq("user_id", current_user_id()).execute()
        st.success(f"✅ 세션 제목을 '{new_title}' 로 갱신했습니다.")
        return True
    except Exception as e:
        st.error(f"제목 보정 실패: {e}")
        return False


def save_documents_to_supabase(
    chunks: List[Any], embeddings: OpenAIEmbeddings, session_id: str
) -> bool:
    if not supabase or not current_user_id():
        return False
    try:
        if not ensure_session_exists(session_id):
            return False

        batch_size = 50
        saved_count = 0
        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i : i + batch_size]

            texts: List[str] = []
            cleaned_chunks: List[Any] = []
            for chunk in batch_chunks:
                clean_text = sanitize_text(chunk.page_content)
                if not clean_text.strip():
                    continue
                meta = (chunk.metadata or {}).copy()
                for k, v in list(meta.items()):
                    if isinstance(v, str):
                        meta[k] = sanitize_text(v)
                chunk.page_content = clean_text
                chunk.metadata = meta
                texts.append(clean_text)
                cleaned_chunks.append(chunk)

            if not texts:
                continue

            batch_embeddings = embeddings.embed_documents(texts)

            documents_to_save = []
            for chunk, embedding in zip(cleaned_chunks, batch_embeddings):
                metadata = chunk.metadata.copy() if chunk.metadata else {}
                metadata["session_id"] = session_id
                metadata["user_id"] = current_user_id()
                content_hash = hashlib.sha256(chunk.page_content.encode("utf-8")).hexdigest()
                documents_to_save.append(
                    {
                        "content": chunk.page_content,
                        "metadata": metadata,
                        "embedding": embedding,
                        "session_id": session_id,
                        "user_id": current_user_id(),
                        "content_hash": content_hash,
                    }
                )

            if documents_to_save:
                try:
                    result = supabase.table("documents").upsert(
                        documents_to_save,
                        on_conflict="session_id,content_hash",
                    ).execute()
                    if result.data:
                        saved_count += len(result.data)
                except Exception as e:
                    msg = str(e).lower()
                    if "row-level security" in msg or "rls" in msg:
                        st.warning("documents upsert 가 거부되었습니다. DB 정책 또는 user_id 값을 확인해주세요.")
                    else:
                        st.warning(f"문서 upsert 중 일부 오류: {e}")

        return saved_count > 0
    except Exception as e:
        st.error(f"문서 저장 중 오류: {e}")
        return False


def generate_followup_questions(
    user_question: str, ai_response: str, context_text: str
) -> List[str]:
    try:
        model_name = st.session_state.selected_model
        if model_name == "gpt-5.5":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                return []
            llm = ChatOpenAI(model="gpt-5.5", temperature=1, openai_api_key=api_key)
        elif model_name == "claude-opus-4-7":
            claude_key = os.getenv("ANTHROPIC_API_KEY")
            if not claude_key:
                return []
            llm = ChatAnthropic(model="claude-opus-4-7", temperature=1, anthropic_api_key=claude_key)
        elif model_name == "gemini-3-pro-preview":
            gemini_key = os.getenv("GOOGLE_API_KEY")
            if not gemini_key:
                return []
            llm = ChatGoogleGenerativeAI(
                model="gemini-3-pro-preview", temperature=1, google_api_key=gemini_key
            )
        else:
            return []

        prompt = f"""다음 질문과 답변을 기반으로, 사용자가 더 깊이 있게 알아볼 수 있는 관련 질문 3개를 생성해주세요.

원래 질문: {user_question}

답변 내용:
{ai_response[:1000]}

관련 문서 컨텍스트:
{context_text[:500]}

요구사항:
- 답변 내용과 관련 문서를 바탕으로 더 깊이 있는 질문 생성
- 각 질문은 한 문장으로 작성
- 질문은 구체적이고 실용적이어야 함
- 질문만 출력 (번호나 설명 없이)
- 각 질문은 줄바꿈으로 구분

관련 질문:"""
        text = llm.invoke(prompt).content.strip()
        questions: List[str] = []
        for line in text.split("\n"):
            line = line.strip()
            if not line:
                continue
            for prefix in ["1.", "2.", "3.", "질문 1:", "질문 2:", "질문 3:", "-", "•"]:
                if line.startswith(prefix):
                    line = line[len(prefix) :].strip()
            if len(line) > 5:
                questions.append(line)
        return questions[:3]
    except Exception:
        return []


def search_with_web_fallback(query: str) -> str:
    """PDF 가 없을 때 OpenAI web_search 로 답변."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        return "사이드바에 OpenAI API Key 를 입력해주세요. (PDF 가 없으면 웹검색으로 답변합니다)"
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        recent_history = st.session_state.chat_history[-6:] if st.session_state.chat_history else []
        history_text = ""
        for msg in recent_history:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role in ("user", "assistant") and content:
                role_kr = "사용자" if role == "user" else "AI"
                history_text += f"{role_kr}: {content}\n"
        prompt = f"""당신은 신뢰할 수 있는 한국어 어시스턴트입니다.
아래 질문에 대해 웹검색을 활용해 최신 정보를 반영하여 답변해주세요.

이전 대화:
{history_text}

현재 질문:
{query}

요구사항:
- 한국어 존댓말로 답변
- 핵심을 먼저 말하고, 필요한 경우 소제목으로 구조화
- 확인이 필요한 내용은 단정하지 않고 신중히 표현
"""
        response = client.responses.create(
            model="gpt-5.5",
            input=prompt,
            tools=[{"type": "web_search", "search_context_size": "high"}],
        )
        answer = (response.output_text or "").strip()
        return answer or "웹검색 결과를 바탕으로 답변을 생성하지 못했습니다."
    except Exception as e:
        return f"웹검색 중 오류가 발생했습니다: {e}"


def get_supabase_status() -> Dict[str, Any]:
    url = os.getenv("SUPABASE_URL")
    key = _get_supabase_anon_key()
    status: Dict[str, Any] = {
        "has_url": bool(url),
        "has_key": bool(key),
        "connected": supabase is not None,
        "logged_in": current_user_id() is not None,
        "query_ok": False,
        "session_query_ok": None,
        "error": None,
    }
    if not supabase:
        return status
    try:
        supabase.table("app_users").select("id").limit(1).execute()
        status["query_ok"] = True
    except Exception as e:
        status["error"] = str(e)
        if is_missing_table_error(e):
            status["schema_mismatch"] = (
                "`app_users` 테이블 없음 — app_users_only.sql 또는 multi-users-ref.sql 실행"
            )
    uid = current_user_id()
    if uid:
        try:
            supabase.table("sessions").select("id").eq("user_id", uid).limit(1).execute()
            status["session_query_ok"] = True
        except Exception as e:
            status["session_query_ok"] = False
            status["error"] = str(e)
            if is_sessions_user_column_missing_error(e):
                status["schema_mismatch"] = (
                    "구버전 DB: sessions.user_id 없음 — migrate_legacy_sessions_user_id.sql 실행"
                )
    return status


@st.cache_data(ttl=30)
def check_required_tables() -> Dict[str, bool]:
    tables = {"sessions": False, "messages": False, "documents": False}
    if not supabase:
        return tables
    for t in tables:
        try:
            supabase.table(t).select("id").limit(1).execute()
            tables[t] = True
        except Exception:
            tables[t] = False
    return tables


# ===========================================================================
# 사이드바
# ===========================================================================
with st.sidebar:
    # 1) LLM API 키 입력 (멀티유저용 - .env 의존 금지)
    st.markdown('<h2 style="color: #1f77b4;">LLM API 키 입력</h2>', unsafe_allow_html=True)
    st.session_state.openai_api_key = st.text_input(
        "OpenAI API Key", value=st.session_state.get("openai_api_key", ""), type="password"
    )
    st.session_state.anthropic_api_key = st.text_input(
        "Anthropic API Key", value=st.session_state.get("anthropic_api_key", ""), type="password"
    )
    st.session_state.gemini_api_key = st.text_input(
        "Google Gemini API Key", value=st.session_state.get("gemini_api_key", ""), type="password"
    )
    apply_api_keys_to_env()
    st.markdown("---")

    # 2) 모델 선택
    st.markdown('<h2 style="color: #1f77b4;">LLM 모델 선택</h2>', unsafe_allow_html=True)
    model_options = ["gpt-5.5", "claude-opus-4-7", "gemini-3-pro-preview"]
    try:
        model_index = model_options.index(st.session_state.selected_model)
    except ValueError:
        model_index = 0
    st.session_state.selected_model = st.selectbox(
        "모델 선택", options=model_options, index=model_index, key="model_selectbox"
    )
    st.markdown("---")

    # 3) 인증 영역
    st.markdown('<h2 style="color: #1f77b4;">사용자 인증</h2>', unsafe_allow_html=True)

    if not supabase:
        st.error(
            "Supabase 가 연결되지 않았습니다. `SUPABASE_URL` 과 "
            "`SUPABASE_ANON_KEY`(또는 `SUPABASE_PUBLISHABLE_KEY`) 환경변수를 확인하세요."
        )
    else:
        if current_user():
            user_login_id = current_user().get("login_id", "")
            st.success(f"로그인됨: {user_login_id}")
            if st.button("🚪 로그아웃", use_container_width=True):
                logout_user()
                st.rerun()
        else:
            with st.form("login_form", clear_on_submit=False):
                login_id = st.text_input("user id")
                password = st.text_input("비밀번호", type="password")
                submitted = st.form_submit_button("🔐 로그인", use_container_width=True)
                if submitted:
                    if login_user(login_id, password):
                        st.success("로그인 되었습니다.")
                        st.rerun()
            if st.button("📝 회원가입", use_container_width=True, type="secondary"):
                # Streamlit Cloud 등에 pages/ 가 없으면 switch_page 대신 다이얼로그로 가입
                if (current_dir / "pages" / "sign_up.py").is_file():
                    st.switch_page("pages/sign_up.py")
                else:
                    open_signup_dialog()

    st.markdown("---")

    # 4) Supabase 상태
    with st.expander("Supabase 상태", expanded=False):
        sb_status = get_supabase_status()
        st.write(f"URL 설정: {'✅' if sb_status['has_url'] else '❌'}")
        st.write(f"KEY 설정: {'✅' if sb_status['has_key'] else '❌'}")
        st.write(f"클라이언트: {'✅' if sb_status['connected'] else '❌'}")
        st.write(f"로그인 상태: {'✅' if sb_status['logged_in'] else '❌'}")
        st.write(f"app_users 접근: {'✅' if sb_status['query_ok'] else '❌'}")
        sq = sb_status.get("session_query_ok")
        if sq is None:
            st.write("내 세션 조회: — (로그인 후 확인)")
        else:
            st.write(f"내 세션 조회: {'✅' if sq else '❌'}")
        if sb_status.get("schema_mismatch"):
            st.info(sb_status["schema_mismatch"])
        if sb_status.get("error"):
            st.warning(f"오류: {sb_status['error']}")
        required = check_required_tables()
        st.write(f"sessions 테이블: {'✅' if required['sessions'] else '❌'}")
        st.write(f"messages 테이블: {'✅' if required['messages'] else '❌'}")
        st.write(f"documents 테이블: {'✅' if required['documents'] else '❌'}")

# ---------------------------------------------------------------------------
# 인증 게이트: 로그인하지 않은 경우 메인 본문에 안내만 출력
# ---------------------------------------------------------------------------
if not supabase:
    st.error(
        "Supabase 연결 정보가 없습니다. Streamlit Secrets 또는 .env 에 "
        "`SUPABASE_URL`, `SUPABASE_ANON_KEY` (또는 `SUPABASE_PUBLISHABLE_KEY`) 를 설정해주세요."
    )
    st.stop()

if not current_user():
    st.info("좌측 사이드바에서 로그인하세요. 처음 사용하시면 '회원가입' 버튼을 눌러주세요.")
    st.stop()

# ---------------------------------------------------------------------------
# 사이드바 (로그인 후 표시) : 세션 관리 / PDF 업로드
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown('<h2 style="color: #1f77b4;">세션 관리</h2>', unsafe_allow_html=True)

    required_tables = check_required_tables()
    has_required_tables = all(required_tables.values())

    selected_session_id: Optional[str] = None
    selected_session_display: str = "새 세션"

    if has_required_tables:
        sessions = get_sessions()
        session_options = ["새 세션"]
        session_map: Dict[str, str] = {}
        kst = timezone(timedelta(hours=9))

        for s in sessions:
            title = s.get("title", "New Chat")
            sid = s.get("id")
            created_at = s.get("created_at", "")
            display_title = title
            if created_at:
                try:
                    dt = datetime.fromisoformat(str(created_at).replace("Z", "+00:00"))
                    display_title = f"{title} ({dt.astimezone(kst).strftime('%m/%d %H:%M')})"
                except Exception:
                    pass
            unique_key = display_title
            counter = 1
            while unique_key in session_map:
                counter += 1
                unique_key = f"{display_title} #{counter}"
            session_options.append(unique_key)
            session_map[unique_key] = sid

        # 현재 세션 인덱스
        current_index = 0
        for idx, s in enumerate(sessions):
            if s["id"] == st.session_state.current_session_id:
                current_index = idx + 1
                break

        selected_session_display = st.selectbox(
            "세션 선택",
            options=session_options,
            index=current_index if current_index < len(session_options) else 0,
            key="session_selectbox",
        )
        if selected_session_display != "새 세션":
            selected_session_id = session_map.get(selected_session_display)

        # 풀다운 변경 시 자동 로드 (현재 세션 자동 저장)
        if (
            selected_session_id
            and selected_session_id != st.session_state.current_session_id
        ):
            if st.session_state.current_session_id and st.session_state.chat_history:
                save_session(st.session_state.current_session_id)
            with st.spinner(f"세션 '{selected_session_display}' 로드 중..."):
                if load_session(selected_session_id):
                    st.session_state.current_session_id = selected_session_id
                    st.success(f"✅ 세션 '{selected_session_display}' 로드 완료")
                    st.rerun()
                else:
                    st.error("❌ 세션 로드에 실패했습니다.")

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                "📂 세션 로드",
                use_container_width=True,
                disabled=(selected_session_display == "새 세션" or selected_session_id is None),
            ):
                if (
                    st.session_state.current_session_id
                    and st.session_state.current_session_id != selected_session_id
                    and st.session_state.chat_history
                ):
                    save_session(st.session_state.current_session_id)
                with st.spinner(f"세션 '{selected_session_display}' 로드 중..."):
                    if load_session(selected_session_id):
                        st.session_state.current_session_id = selected_session_id
                        st.success(f"✅ 세션 '{selected_session_display}' 로드 완료")
                        st.rerun()
                    else:
                        st.error("❌ 세션 로드에 실패했습니다.")
        with col2:
            if st.button("➕ 새 세션", use_container_width=True):
                if st.session_state.current_session_id and st.session_state.chat_history:
                    save_session(st.session_state.current_session_id)
                new_id = create_session()
                if new_id:
                    st.session_state.current_session_id = new_id
                    st.session_state.chat_history = []
                    st.session_state.conversation_memory = []
                    st.session_state.processed_files = []
                    st.session_state.retriever = None
                    st.success("✅ 새 세션이 생성되었습니다.")
                    st.rerun()

        col_save, col_delete = st.columns(2)
        with col_save:
            if st.button("💾 세션 저장", use_container_width=True):
                if st.session_state.current_session_id:
                    if save_session(st.session_state.current_session_id):
                        st.session_state.chat_history = []
                        st.session_state.conversation_memory = []
                        st.session_state.processed_files = []
                        st.session_state.retriever = None
                        st.session_state.current_session_id = str(uuid.uuid4())
                        st.success("✅ 세션 저장 후 화면을 초기화했습니다.")
                        st.rerun()
        with col_delete:
            if selected_session_display != "새 세션" and selected_session_id:
                if st.button("🗑️ 세션 삭제", use_container_width=True, type="secondary"):
                    with st.spinner("세션 삭제 중..."):
                        if delete_session(selected_session_id):
                            st.success(f"✅ 세션 '{selected_session_display}' 삭제 완료")
                            if selected_session_id == st.session_state.current_session_id:
                                new_id = create_session()
                                st.session_state.current_session_id = new_id or str(uuid.uuid4())
                                st.session_state.chat_history = []
                                st.session_state.conversation_memory = []
                                st.session_state.processed_files = []
                                st.session_state.retriever = None
                            st.rerun()
                        else:
                            st.error("❌ 세션 삭제 실패")
            else:
                st.button("🗑️ 세션 삭제", use_container_width=True, disabled=True, type="secondary")

        col_clear, col_fix = st.columns(2)
        with col_clear:
            if st.button("🔄 화면 초기화", use_container_width=True):
                st.session_state.chat_history = []
                st.session_state.conversation_memory = []
                st.session_state.processed_files = []
                st.session_state.retriever = None
                st.success("✅ 화면이 초기화되었습니다.")
                st.rerun()
        with col_fix:
            if st.button("✏️ 제목보정", use_container_width=True):
                if st.session_state.current_session_id:
                    if fix_session_title(st.session_state.current_session_id):
                        st.rerun()

        if st.button("🗂️ vectordb", use_container_width=True):
            sources = set()
            try:
                doc_res = (
                    supabase.table("documents")
                    .select("metadata")
                    .eq("session_id", st.session_state.current_session_id)
                    .eq("user_id", current_user_id())
                    .execute()
                )
                if doc_res.data:
                    for d in doc_res.data:
                        meta = d.get("metadata", {}) or {}
                        src = meta.get("source")
                        if src:
                            sources.add(str(src))
            except Exception as e:
                st.error(f"벡터 DB 조회 실패: {e}")
            if st.session_state.processed_files:
                for f in st.session_state.processed_files:
                    sources.add(str(f))
            if sources:
                st.info("현재 세션 파일 목록:\n" + "\n".join(sorted(sources)))
            else:
                st.warning("현재 세션에서 확인된 파일이 없습니다.")
    else:
        st.warning(
            "Supabase 연결은 되었지만 필수 테이블이 없습니다. "
            "`multi-users-ref.sql` 을 SQL Editor 에서 먼저 실행하세요."
        )

    st.markdown("---")
    st.markdown('<h2 style="color: #1f77b4;">PDF 파일 업로드</h2>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "PDF 파일을 선택하세요", type="pdf", accept_multiple_files=True
    )

    if uploaded_files and st.button("파일 처리하기"):
        with st.spinner("PDF 파일을 처리 중입니다..."):
            try:
                temp_dir = tempfile.TemporaryDirectory()
                all_docs: List[Document] = []
                new_files: List[str] = []
                for uploaded_file in uploaded_files:
                    if uploaded_file.name in st.session_state.processed_files:
                        continue
                    temp_file_path = os.path.join(temp_dir.name, uploaded_file.name)
                    with open(temp_file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    documents = PyPDFLoader(temp_file_path).load()
                    for doc in documents:
                        doc.metadata["source"] = uploaded_file.name
                    all_docs.extend(documents)
                    new_files.append(uploaded_file.name)

                if not all_docs:
                    st.success("모든 파일이 이미 처리되었습니다.")
                else:
                    text_splitter = RecursiveCharacterTextSplitter(
                        chunk_size=500, chunk_overlap=100, length_function=len
                    )
                    raw_chunks = text_splitter.split_documents(all_docs)
                    chunks = []
                    for chunk in raw_chunks:
                        clean_text = sanitize_text(chunk.page_content)
                        if not clean_text.strip():
                            continue
                        meta = (chunk.metadata or {}).copy()
                        for k, v in list(meta.items()):
                            if isinstance(v, str):
                                meta[k] = sanitize_text(v)
                        chunk.page_content = clean_text
                        chunk.metadata = meta
                        chunks.append(chunk)

                    if not chunks:
                        st.error(
                            "텍스트를 추출한 청크가 없습니다. (빈 PDF, 스캔본만 있는 PDF 등) "
                            "다른 파일을 업로드하거나, 검색 가능한 PDF로 준비해 주세요."
                        )
                    else:
                        total_chunks = len(chunks)
                        st.info(f"총 {total_chunks}개의 청크를 처리합니다.")

                        api_key = os.getenv("OPENAI_API_KEY")
                        if not api_key:
                            st.error("사이드바에 OpenAI API Key 를 입력해주세요.")
                        else:
                            embeddings = OpenAIEmbeddings(openai_api_key=api_key)
                            retriever_ready = False
                            backend = None

                            ok = save_documents_to_supabase(
                                chunks, embeddings, st.session_state.current_session_id
                            )
                            if ok:
                                st.success(f"✅ {total_chunks}개 청크 저장 완료 (Supabase)")
                                try:
                                    st.session_state.retriever = SessionRetriever(
                                        supabase,
                                        embeddings,
                                        st.session_state.current_session_id,
                                        k=10,
                                    )
                                    retriever_ready = True
                                    backend = "supabase"
                                except Exception as e:
                                    st.warning(f"Supabase Retriever 초기화 실패: {e}")
                            else:
                                st.warning("Supabase 저장 실패. 로컬 FAISS 로 대체합니다.")

                            if not retriever_ready:
                                try:
                                    k = min(10, len(chunks))
                                    vectorstore = FAISS.from_documents(chunks, embeddings)
                                    st.session_state.vectorstore = vectorstore
                                    st.session_state.retriever = vectorstore.as_retriever(
                                        search_kwargs={"k": k}
                                    )
                                    retriever_ready = True
                                    backend = "local_faiss"
                                    st.success(
                                        f"✅ 로컬 벡터스토어로 검색 준비 완료 (k={k})"
                                    )
                                except Exception as e:
                                    st.error(f"로컬 벡터스토어 생성 실패: {e}")

                            if retriever_ready:
                                st.info(f"검색 백엔드: {backend}")
                                st.session_state.processed_files.extend(new_files)
                                save_session(st.session_state.current_session_id)
                                st.success("파일이 처리되었고 세션이 자동 저장되었습니다.")
            except Exception as e:
                st.error(f"파일 처리 중 오류: {e}")

    if st.session_state.processed_files:
        st.markdown(
            '<h3 style="color: #ffd700;">처리된 파일 목록</h3>',
            unsafe_allow_html=True,
        )
        for f in st.session_state.processed_files:
            st.write(f"- {f}")
        st.subheader("📊 시스템 상태")
        st.info(f"처리된 파일 수: {len(st.session_state.processed_files)}")
        st.info(f"대화 기록 수: {len(st.session_state.chat_history)}")


# ---------------------------------------------------------------------------
# 앱 시작 시: 가장 최근 세션 자동 로드
# ---------------------------------------------------------------------------
if (
    supabase
    and current_user()
    and all(check_required_tables().values())
    and not st.session_state.sessions_loaded
):
    sessions = get_sessions()
    if sessions:
        latest_session_id = sessions[0]["id"]
        if load_session(latest_session_id):
            st.session_state.current_session_id = latest_session_id
    st.session_state.sessions_loaded = True


# ---------------------------------------------------------------------------
# 채팅 화면
# ---------------------------------------------------------------------------
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    if st.session_state.retriever is None:
        with st.chat_message("assistant"):
            with st.spinner("PDF 가 없어 웹검색으로 답변을 생성하는 중입니다..."):
                response = search_with_web_fallback(prompt)
                st.write(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})
        st.session_state.conversation_memory.append(f"사용자: {prompt}")
        st.session_state.conversation_memory.append(f"AI: {response}")
        if len(st.session_state.conversation_memory) > 100:
            st.session_state.conversation_memory = st.session_state.conversation_memory[-100:]
        save_session(st.session_state.current_session_id)
    else:
        try:
            retrieved_docs = st.session_state.retriever.invoke(prompt)
            if not retrieved_docs:
                response = f"죄송합니다. '{prompt}' 에 대한 관련 문서를 찾을 수 없습니다."
                with st.chat_message("assistant"):
                    st.write(response)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            else:
                top_docs = retrieved_docs[:3]
                context_text = ""
                max_context_length = 8000
                current_length = 0
                for i, doc in enumerate(top_docs):
                    doc_text = f"[문서 {i+1}]\n{doc.page_content}\n\n"
                    if current_length + len(doc_text) > max_context_length:
                        st.warning(f"토큰 제한으로 인해 문서 {i+1}개만 사용합니다.")
                        break
                    context_text += doc_text
                    current_length += len(doc_text)

                conversation_context = ""
                if st.session_state.conversation_memory:
                    conversation_context = "\n\n=== 이전 대화 맥락 ===\n"
                    for conv in st.session_state.conversation_memory[-50:]:
                        conversation_context += f"{conv}\n"
                    conversation_context += "=== 대화 맥락 끝 ===\n"

                system_prompt = f"""
질문: {prompt}

관련 문서:
{context_text}{conversation_context}

위 문서 내용과 이전 대화 맥락을 모두 고려하여 질문에 답변해주세요.
이전 대화에서 언급된 내용이 있다면 그것을 참조하여 더 정확하고 맥락적인 답변을 제공하세요.

답변 형식:
- 답변은 반드시 헤딩(# ## ###)을 사용하여 구조화하세요
- 주요 주제는 # (H1)로, 세부 내용은 ## (H2)로, 구체적 설명은 ### (H3)로 구분하세요
- 답변은 서술형으로 작성하되 존대말을 사용하세요
- 개조식이나 불완전한 문장을 사용하지 말고, 완전한 문장으로 서술하세요

주의사항:
- 답변 중간에 (문서1), (문서2) 같은 참조 표시를 하지 마세요
- "참조 문서:", "제공된 문서", "문서 1, 문서 2" 같은 문구를 사용하지 마세요
- 답변은 순수한 내용만 포함하고, 참조 관련 문구는 전혀 포함하지 마세요
- 답변 끝에 참조 정보나 출처 관련 문구를 추가하지 마세요
"""

                model_name = st.session_state.selected_model
                openai_key = os.getenv("OPENAI_API_KEY")

                if model_name == "gpt-5.5":
                    if not openai_key:
                        st.error("사이드바에 OpenAI API Key 를 입력해주세요.")
                        st.stop()
                    llm = ChatOpenAI(
                        model="gpt-5.5", temperature=1, openai_api_key=openai_key, streaming=True
                    )
                elif model_name == "claude-opus-4-7":
                    claude_key = os.getenv("ANTHROPIC_API_KEY")
                    if not claude_key:
                        st.error("사이드바에 Anthropic API Key 를 입력해주세요.")
                        st.stop()
                    llm = ChatAnthropic(
                        model="claude-opus-4-7",
                        temperature=1,
                        anthropic_api_key=claude_key,
                        streaming=True,
                    )
                elif model_name == "gemini-3-pro-preview":
                    gemini_key = os.getenv("GOOGLE_API_KEY")
                    if not gemini_key:
                        st.error("사이드바에 Google Gemini API Key 를 입력해주세요.")
                        st.stop()
                    llm = ChatGoogleGenerativeAI(
                        model="gemini-3-pro-preview",
                        temperature=1,
                        google_api_key=gemini_key,
                        streaming=True,
                    )
                else:
                    if not openai_key:
                        st.error("사이드바에 OpenAI API Key 를 입력해주세요.")
                        st.stop()
                    llm = ChatOpenAI(
                        model="gpt-5.5", temperature=1, openai_api_key=openai_key, streaming=True
                    )

                with st.chat_message("assistant"):
                    response_placeholder = st.empty()
                    full_response = ""
                    for chunk in llm.stream(system_prompt):
                        content = getattr(chunk, "content", str(chunk))
                        if content:
                            full_response += content
                            response_placeholder.write(full_response + "▌")

                    followup_questions = generate_followup_questions(
                        prompt, full_response, context_text
                    )
                    if followup_questions:
                        full_response += "\n\n---\n\n### 💡 더 알아보기\n\n다음 질문들도 도움이 될 수 있습니다:\n\n"
                        for i, q in enumerate(followup_questions, 1):
                            full_response += f"{i}. {q}\n"

                    response_placeholder.write(full_response)

                response = full_response
                st.session_state.chat_history.append({"role": "assistant", "content": response})
                st.session_state.conversation_memory.append(f"사용자: {prompt}")
                st.session_state.conversation_memory.append(f"AI: {response}")
                if len(st.session_state.conversation_memory) > 100:
                    st.session_state.conversation_memory = st.session_state.conversation_memory[-100:]

                save_session(st.session_state.current_session_id)

        except Exception as e:
            with st.chat_message("assistant"):
                st.write(f"오류가 발생했습니다: {e}")
            st.session_state.chat_history.append(
                {"role": "assistant", "content": f"오류가 발생했습니다: {e}"}
            )
