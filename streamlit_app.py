"""
Very basic web UI for the Travel AI Agent — no predefined user profile.
Every browser session starts a completely fresh conversation; the agent
learns the client's needs (origin city, preferences, etc.) purely from
what they say in the chat.

Run with:
    streamlit run streamlit_app.py
"""
import os
import uuid
import streamlit as st

# ── Bridge Streamlit Cloud "secrets" into real environment variables ──────
# main.py reads its config with os.getenv(...). Locally that comes from a
# .env file (via load_dotenv()). On Streamlit Community Cloud there is no
# .env file — config is set instead in the app's "Secrets" panel and only
# shows up as st.secrets. This copies each secret into os.environ *before*
# main.py is imported, so os.getenv("OPENROUTER_API_KEY") etc. still work
# unchanged in both places.
for _k, _v in st.secrets.items():
    os.environ[_k] = str(_v)

from langchain_core.messages import HumanMessage, AIMessage
from main import app

st.set_page_config(page_title="Travel Agent", page_icon="✈️")

st.title("✈️ Travel AI Agent")
st.caption("Database-powered · Your Friend as a Travel Agent · Ready to Serve · zero hallucination")

# Fresh thread + fresh chat history every time the page loads / is refreshed
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []  # list of ("user"/"assistant", text)

# Render past messages
for role, text in st.session_state.chat_history:
    with st.chat_message(role):
        st.markdown(text)

# Chat input
user_input = st.chat_input("Ask me about flights, hotels, attractions, or a full trip plan...")

if user_input:
    st.session_state.chat_history.append(("user", user_input))
    with st.chat_message("user"):
        st.markdown(user_input)

    config = {"configurable": {"thread_id": st.session_state.thread_id}}
    input_state = {"messages": [HumanMessage(content=user_input)]}

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            for _ in app.stream(input_state, config=config):
                pass
            current_state = app.get_state(config)
            # SAFETY: only ever show the user a real AIMessage. If something
            # upstream failed and left an internal SystemMessage/ToolMessage
            # as the last item, never display it raw — that is exactly what
            # caused internal "BRIEFING —..." text to leak into the chat.
            last_msg = current_state.values["messages"][-1]
            if isinstance(last_msg, AIMessage):
                reply = last_msg.content
            else:
                reply = (
                    "Sorry — I ran into a technical hiccup putting that answer together. "
                    "Could you try asking that again?"
                )
        st.markdown(reply)

    st.session_state.chat_history.append(("assistant", reply))

# Note: Streamlit keeps session_state alive across reruns AND across a
# plain browser refresh in the same tab (it reuses the same session).
# Use the "Start a new conversation" button below, or open a new tab /
# incognito window, to actually get a fresh session_state and thread_id.
if st.session_state.chat_history:
    if st.button("🔄 Start a new conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat_history = []
        st.rerun()
