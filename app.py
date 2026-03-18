import streamlit as st
from rag_engine import (
    load_file_bytes,
    chunk_text,
    create_index,
    get_embed_model,
    process_question,
)

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="ExamPrep AI", page_icon="📚", layout="centered")

# ── Styling ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Space Grotesk', sans-serif; }
    .stApp { background-color: #0f1117; color: #e8eaf0; }
    .title-block { text-align: center; padding: 2rem 0 1rem 0; }
    .title-block h1 {
        font-size: 2.8rem; font-weight: 700;
        background: linear-gradient(135deg, #6ee7f7, #a78bfa);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .title-block p { color: #9ca3af; font-size: 1rem; }
    .chat-message { padding: 1rem 1.2rem; border-radius: 10px; margin-bottom: 0.8rem; line-height: 1.6; }
    .user-msg { background: #1e2235; border-left: 3px solid #a78bfa; }
    .assistant-msg { background: #162230; border-left: 3px solid #6ee7f7; }
    .status-badge {
        display: inline-block; background: #162a1e; color: #4ade80;
        border: 1px solid #166534; border-radius: 20px;
        padding: 0.25rem 0.8rem; font-size: 0.8rem; font-weight: 600;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6ee7f7, #a78bfa);
        color: #0f1117; font-weight: 700; border: none; border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
for key, default in {
    "index": None, "chunks": [], "chat_history": [],
    "messages": [], "file_loaded": False
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Controls")

    if st.button("🗑️ Clear Chat"):
        st.session_state.messages = []
        st.session_state.chat_history = []
        st.rerun()

    if st.button("📄 Reset Document"):
        for key in ["chunks", "messages", "chat_history"]:
            st.session_state[key] = []
        st.session_state.index = None
        st.session_state.file_loaded = False
        st.rerun()

    if st.session_state.file_loaded:
        st.markdown(
            f'<div class="status-badge">✅ {len(st.session_state.chunks)} chunks loaded</div>',
            unsafe_allow_html=True
        )

# ── Title ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="title-block">
    <h1>📚 ExamPrep AI</h1>
    <p>Upload your study material and ask anything — get exam-ready answers instantly</p>
</div>
""", unsafe_allow_html=True)

# ── File Upload ───────────────────────────────────────────────────────────────
if not st.session_state.file_loaded:
    uploaded_file = st.file_uploader("Upload your study material", type=["pdf", "docx"])

    if uploaded_file:
        with st.spinner("📄 Processing your document..."):
            get_embed_model()  # Warm up the embedding model
            text = load_file_bytes(uploaded_file.read(), uploaded_file.name)

            if text.strip():
                chunks = chunk_text(text)
                index = create_index(chunks)
                st.session_state.chunks = chunks
                st.session_state.index = index
                st.session_state.file_loaded = True
                st.success(f"✅ Ready! Loaded {len(chunks)} chunks.")
                st.rerun()
            else:
                st.error("Could not extract text from file.")

# ── Chat Interface ────────────────────────────────────────────────────────────
if st.session_state.file_loaded:
    for msg in st.session_state.messages:
        css = "user-msg" if msg["role"] == "user" else "assistant-msg"
        icon = "🧑‍🎓" if msg["role"] == "user" else "🤖"
        st.markdown(
            f'<div class="chat-message {css}"><strong>{icon}</strong><br>{msg["content"]}</div>',
            unsafe_allow_html=True
        )

    question = st.chat_input("Ask a question about your study material...")

    if question:
        st.session_state.messages.append({"role": "user", "content": question})

        with st.spinner("⏳ Finding answer..."):
            try:
                answer = process_question(
                    question,
                    st.session_state.index,
                    st.session_state.chunks,
                    st.session_state.chat_history,
                )
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.chat_history.extend([
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": answer}
                ])
                if len(st.session_state.chat_history) > 12:
                    st.session_state.chat_history = st.session_state.chat_history[-12:]
                st.rerun()
            except Exception as e:
                st.error(f"❌ Error: {e}")