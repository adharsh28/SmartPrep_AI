import faiss
import numpy as np
import re
import os
import tempfile
from sentence_transformers import SentenceTransformer
from pypdf import PdfReader
from groq import Groq
from dotenv import load_dotenv

load_dotenv()  # loads from .env file if present

try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    DOCX_AVAILABLE = False

GROQ_MODEL = "llama-3.3-70b-versatile"

# ── Groq client — initialized once from env ───────────────────────────────────
_groq_client = None

def get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "GROQ_API_KEY not set. Add it to your .env file or environment variables."
            )
        _groq_client = Groq(api_key=api_key)
    return _groq_client


# ── Embedding model — loaded once ─────────────────────────────────────────────
_embed_model = None

def get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer('all-MiniLM-L6-v2')
    return _embed_model


# ── File Loading ──────────────────────────────────────────────────────────────
def load_file(file_path: str) -> str:
    """Load text from a PDF or Word (.docx) file path."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        return "\n".join(p.extract_text() or "" for p in reader.pages)

    elif ext == ".docx":
        if not DOCX_AVAILABLE:
            raise ImportError("Install python-docx: pip install python-docx")
        doc = Document(file_path)
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

    else:
        raise ValueError(f"Unsupported file type: {ext}. Use .pdf or .docx")


def load_file_bytes(file_bytes: bytes, filename: str) -> str:
    """Load text from raw bytes — used by Streamlit file uploader."""
    ext = os.path.splitext(filename)[1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    try:
        return load_file(tmp_path)
    finally:
        os.unlink(tmp_path)


# ── Chunking ──────────────────────────────────────────────────────────────────
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list:
    """Split text into overlapping chunks."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks, current = [], ""
    for sentence in sentences:
        if len(current) + len(sentence) <= chunk_size:
            current += " " + sentence
        else:
            if current.strip():
                chunks.append(current.strip())
            current = current[-overlap:] + " " + sentence if len(current) > overlap else sentence
    if current.strip():
        chunks.append(current.strip())
    return chunks


# ── FAISS Index ───────────────────────────────────────────────────────────────
def create_index(chunks: list):
    """Embed chunks and build a FAISS index."""
    embeddings = get_embed_model().encode(chunks, show_progress_bar=False).astype("float32")
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    return index


# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query: str, index, chunks: list, k: int = 4) -> list:
    """Return the top-k most relevant chunks for a query."""
    query_vec = get_embed_model().encode([query]).astype("float32")
    _, indices = index.search(query_vec, k=k)
    return [chunks[i] for i in indices[0] if i < len(chunks)]


# ── Query Rewriting ───────────────────────────────────────────────────────────
def rewrite_query(question: str, chat_history: list) -> str:
    """Rewrite a follow-up question into a standalone search query."""
    if not chat_history:
        return question

    followup_signals = ["it", "this", "that", "them", "they", "more", "example",
                        "explain", "elaborate", "again", "what about", "how about",
                        "code", "show", "give", "tell"]
    is_followup = len(question.split()) < 10 or any(w in question.lower() for w in followup_signals)

    if not is_followup:
        return question

    history_text = "\n".join(
        f"{'Student' if m['role'] == 'user' else 'Assistant'}: {m['content'][:300]}"
        for m in chat_history[-6:]
    )

    response = get_groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {
                "role": "system",
                "content": (
                    "You rewrite follow-up questions into standalone search queries. "
                    "Output ONLY the rewritten query — no explanation, no quotes, no preamble."
                )
            },
            {
                "role": "user",
                "content": (
                    f"Conversation so far:\n{history_text}\n\n"
                    f"Follow-up question: {question}\n\n"
                    f"Rewrite as a standalone, specific search query:"
                )
            }
        ],
        temperature=0.1,
        max_tokens=80,
    )

    rewritten = response.choices[0].message.content.strip()
    return rewritten if rewritten else question


# ── Answer Generation ─────────────────────────────────────────────────────────
def generate_answer(context: str, question: str, chat_history: list) -> str:
    """Send context + question to Groq and return the answer."""
    system_prompt = (
        "You are a helpful exam preparation assistant for students. "
        "Your job is to:\n"
        "1. Answer clearly and in a structured, easy-to-understand way\n"
        "2. Use bullet points or numbered lists when explaining multiple concepts\n"
        "3. Highlight key terms and definitions\n"
        "4. Give examples if it helps understanding\n"
        "5. Only use the provided context — if it's not there, say so honestly\n"
        "Think like a good teacher, not a textbook."
    )

    user_message = (
        f"Study material context:\n\n{context}\n\n---\n\n"
        f"My question: {question}\n\n"
        f"Please explain this clearly so I can understand it for my exam."
    )

    messages = [{"role": "system", "content": system_prompt}]
    if chat_history:
        messages.extend(chat_history[-10:])
    messages.append({"role": "user", "content": user_message})

    response = get_groq_client().chat.completions.create(
        model=GROQ_MODEL,
        messages=messages,
        temperature=0.3,
        max_tokens=1024,
    )

    return response.choices[0].message.content


# ── Full Pipeline ─────────────────────────────────────────────────────────────
def process_question(question: str, index, chunks: list, chat_history: list) -> str:
    """Full RAG pipeline: rewrite query → retrieve chunks → generate answer."""
    search_query = rewrite_query(question, chat_history)
    relevant_chunks = retrieve(search_query, index, chunks, k=4)
    context = "\n\n---\n\n".join(relevant_chunks)
    return generate_answer(context, question, chat_history)


# ── CLI Entry Point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    file_path = input("Enter path to your PDF or Word file: ").strip()

    print("\nLoading file...")
    text = load_file(file_path)
    if not text.strip():
        print("Could not extract text.")
        exit(1)

    chunks = chunk_text(text)
    print(f"   {len(chunks)} chunks created")

    index = create_index(chunks)
    print("Ready! Type 'quit' to exit, 'clear' to reset history.\n")

    chat_history = []

    while True:
        question = input("Your question: ").strip()
        if question.lower() in ("quit", "exit", "q"):
            break
        if question.lower() == "clear":
            chat_history = []
            print("Chat history cleared.\n")
            continue
        if not question:
            continue

        try:
            answer = process_question(question, index, chunks, chat_history)
            print(f"\nAnswer:\n{answer}\n")
            print("-" * 60 + "\n")
            chat_history.extend([
                {"role": "user", "content": question},
                {"role": "assistant", "content": answer}
            ])
            if len(chat_history) > 12:
                chat_history = chat_history[-12:]
        except Exception as e:
            print(f"Error: {e}")