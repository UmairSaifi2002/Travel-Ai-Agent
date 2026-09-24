"""
streamlit_app.py — Streamlit Cloud entrypoint for the Travel AI Agent.

On startup:
  1. Rebuilds travel.db from db.json if it's missing (Streamlit Cloud wipes
     the filesystem on every reboot).
  2. Bridges Streamlit secrets into environment variables so main.py can
     read OPENROUTER_API_KEY etc. via os.getenv().
  3. Imports the agent and renders the chat UI.
"""
import os
import uuid
import streamlit as st

# ── 1. Rebuild travel.db from db.json if missing ─────────────────────────
if not os.path.exists("travel.db"):
    try:
        from migrate_to_sqlite import migrate
        migrate()
    except Exception as e:
        st.error(f"❌ Could not build travel.db: {e}")
        st.stop()

# ── 2. Bridge Streamlit secrets → environment variables ──────────────────
# main.py reads os.getenv("OPENROUTER_API_KEY") etc. On Streamlit Cloud there
# is no .env file, so we copy the secrets here BEFORE importing main.
SECRET_KEYS = (
    "LLM_PROVIDER",
    "OPENROUTER_API_KEY",
    "OPENROUTER_MODEL",
    "GROQ_API_KEY",
    "OPENAI_API_KEY",
    "GEMINI_API_KEY",
)
try:
    for key in SECRET_KEYS:
        if key in st.secrets and not os.getenv(key):
            os.environ[key] = str(st.secrets[key])
except Exception:
    # st.secrets raises when no secrets are configured. Locally we use .env,
    # so this is fine.
    pass

# ── 3. Fail-fast if the API key is missing ──────────────────────────────
st.set_page_config(page_title="Travel Agent", page_icon="✈️")
st.title("✈️ Travel AI Agent")
st.caption("Database-powered · Zero hallucination")

provider = os.getenv("LLM_PROVIDER", "openrouter").lower()
if provider == "openrouter" and not os.getenv("OPENROUTER_API_KEY"):
    st.error(
        "🔑 **API key not configured.**\n\n"
        "In your Streamlit Cloud app: **⋮ → Settings → Secrets**. Paste:\n\n"
        "```toml\n"
        'LLM_PROVIDER = "openrouter"\n'
        'OPENROUTER_API_KEY = "sk-or-v1-..."\n'
        'OPENROUTER_MODEL = "nvidia/nemotron-3-ultra-550b-a55b:free"\n'
        "```\n\n"
        "Then click **Reboot app** (top-right menu)."
    )
    st.stop()

# ── 4. Import the agent (after env is ready) ────────────────────────────
from langchain_core.messages import HumanMessage, AIMessage
from main import app

# ── 5. Session state ─────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

# ── 6. Chat input ────────────────────────────────────────────────────────
user_input = st.chat_input("Ask me about flights, hotels, or a full trip plan...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {
        "configurable": {"thread_id": st.session_state.thread_id},
        "recursion_limit": 30,
    }

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                for _ in app.stream(
                    {"messages": [HumanMessage(content=user_input)]},
                    config=config,
                ):
                    pass
                last = app.get_state(config).values["messages"][-1]
                reply = last.content if isinstance(last, AIMessage) else \
                        "Sorry — something went wrong. Try again."
            except Exception as e:
                reply = f"⚠️ **Error:** `{type(e).__name__}: {e}`"
                print(f"[app error] {type(e).__name__}: {e}")
        st.markdown(reply)

    st.session_state.chat_history.append(("assistant", reply))

# ── 7. Reset button ──────────────────────────────────────────────────────
if st.session_state.chat_history:
    if st.button("🔄 New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()
