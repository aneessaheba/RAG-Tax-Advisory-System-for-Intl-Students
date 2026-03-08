"""
FastAPI web server for the RAG Tax Advisor.
Serves the chat UI and handles /chat and /feedback API requests.

Run with:
    uvicorn server:app --reload
Then open:
    http://localhost:8000
"""
import os
import json
import time
from datetime import datetime

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
import chromadb
from google import genai
from supabase import create_client, Client
from retriever import HybridRetriever
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_client import Counter, Histogram

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
CHROMA_DIR = os.path.join(BASE_DIR, 'tax_rag_data', 'data_work', 'chroma_db')
QUERY_LOG_PATH = os.path.join(BASE_DIR, 'query_log.jsonl')
FEEDBACK_LOG_PATH = os.path.join(BASE_DIR, 'feedback_log.jsonl')

# Supabase client — None if env vars are not set (falls back to local files)
_supabase_url = os.environ.get("SUPABASE_URL", "")
_supabase_key = os.environ.get("SUPABASE_KEY", "")
supabase: Client | None = (
    create_client(_supabase_url, _supabase_key)
    if _supabase_url and _supabase_key
    else None
)
COLLECTION_NAME = "tax_docs"
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.70
RETRY_ATTEMPTS = 3
RETRY_BASE_DELAY = 2

TAX_KEYWORDS = {
    "tax", "taxes", "form", "irs", "filing", "file", "income", "deduction",
    "refund", "w-2", "w2", "1040", "8843", "8233", "1098", "withholding",
    "treaty", "visa", "f-1", "f1", "j-1", "j1", "opt", "cpt", "fica",
    "ssn", "itin", "scholarship", "stipend", "wage", "wages", "salary",
    "resident", "nonresident", "return", "credit", "exemption", "alien",
    "substantial", "presence", "deadline", "april", "extension", "state",
    "federal", "social security", "medicare", "fellowship", "tuition",
}

app = FastAPI()

# Prometheus metrics
Instrumentator().instrument(app).expose(app)

retrieval_latency_metric = Histogram(
    "rag_retrieval_latency_seconds",
    "Hybrid retrieval latency (BM25 + vector + RRF)",
    buckets=[0.05, 0.1, 0.2, 0.3, 0.5, 1.0],
)
llm_latency_metric = Histogram(
    "rag_llm_latency_seconds",
    "LLM generation latency",
    buckets=[0.5, 1.0, 2.0, 3.0, 5.0, 10.0],
)
confidence_metric = Histogram(
    "rag_retrieval_confidence",
    "Retrieval confidence scores",
    buckets=[0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 1.0],
)
fallback_counter = Counter("rag_llm_fallback_total", "Times extractive fallback was used")
refused_counter = Counter("rag_refused_total", "Refused queries", ["reason"])
local_llm_counter = Counter("rag_local_llm_total", "Local LLM calls", ["status"])

# Load ChromaDB + retriever once at startup (not per request)
db_client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = db_client.get_collection(name=COLLECTION_NAME)
retriever = HybridRetriever(collection)

# Serve static files (HTML, CSS, JS)
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")


# --- Request models ---
class ChatRequest(BaseModel):
    question: str
    student_info: dict


class FeedbackRequest(BaseModel):
    question: str
    answer: str
    rating: int  # 1 = helpful, 0 = not helpful


# --- Helper functions ---
def estimate_tokens(text):
    return len(text) // 4


def is_tax_question(question):
    q_lower = question.lower()
    return any(kw in q_lower for kw in TAX_KEYWORDS)


def build_query(student_info, question):
    parts = [
        question,
        f"{student_info['visa_type']} student",
        f"from {student_info['home_country']}",
        f"tax year {student_info['tax_year']}",
    ]
    if any("OPT" in t or "CPT" in t for t in student_info.get('income_types', [])):
        parts.append("OPT CPT employment")
    return " ".join(parts)


def format_context(chunks):
    parts = []
    for c in chunks:
        meta = c["metadata"]
        source = f"[{meta.get('title', 'Unknown')} - p.{meta.get('page_number', '?')}]"
        parts.append(f"{source}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def extractive_fallback(chunks):
    lines = ["Gemini unavailable — showing source excerpts:\n"]
    for i, c in enumerate(chunks[:2], 1):
        meta = c["metadata"]
        source = f"{meta.get('title', 'Unknown')} (p.{meta.get('page_number', '?')})"
        lines.append(f"Source {i}: {source}\n{c['text'].strip()}")
    return "\n\n".join(lines)


def ask_ollama(prompt: str):
    """
    Attempt local LLM inference via Ollama (LLaMA 3.2).
    Returns (answer, success_bool).
    Falls back gracefully if Ollama is not running.
    """
    try:
        import ollama
        response = ollama.generate(
            model="llama3.2",
            prompt=prompt,
            stream=False,
            options={"temperature": 0.1, "num_predict": 512},
        )
        return response["response"].strip(), True
    except Exception:
        return None, False


def ask_gemini(student_info, context, question, chunks):
    prompt = f"""You are a helpful tax advisor for international students in the U.S.

Student profile:
- Visa: {student_info['visa_type']}
- Home country: {student_info['home_country']}
- First U.S. entry: {student_info['first_entry_year']}
- Tax year: {student_info['tax_year']}
- Income types: {', '.join(student_info.get('income_types', ['None']))}
- State: {student_info['state']}
- Has SSN/ITIN: {'Yes' if student_info.get('has_ssn_or_itin') else 'No'}

Use ONLY the provided reference documents to answer. If the documents don't cover something,
say so clearly. Always remind the student this is general guidance, not professional tax advice.

--- REFERENCE DOCUMENTS ---
{context}
--- END DOCUMENTS ---

Student's question: {question}

Provide a clear, helpful answer:"""

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    t0 = time.time()

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        if attempt > 1:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 2))
            time.sleep(delay)
        try:
            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            latency = round(time.time() - t0, 2)
            return response.text, latency, estimate_tokens(prompt), estimate_tokens(response.text), False
        except Exception:
            pass

    latency = round(time.time() - t0, 2)
    return extractive_fallback(chunks), latency, 0, 0, True


def ask_llm_with_routing(student_info, context, question, chunks):
    """
    Model routing: try local LLaMA (Ollama) first, fall back to Gemini.
    Routing logic:
      1. Build prompt
      2. Try Ollama (local LLaMA 3.2) — fast, free, private
      3. If Ollama unavailable/fails -> route to Gemini 2.0 Flash
    Returns (answer, latency, input_tokens, output_tokens, used_fallback, model_used)
    """
    t0 = time.time()

    prompt = f"""You are a helpful tax advisor for international students in the U.S.

Student profile:
- Visa: {student_info['visa_type']}
- Home country: {student_info['home_country']}
- First U.S. entry: {student_info['first_entry_year']}
- Tax year: {student_info['tax_year']}
- Income types: {', '.join(student_info.get('income_types', ['None']))}
- State: {student_info['state']}
- Has SSN/ITIN: {'Yes' if student_info.get('has_ssn_or_itin') else 'No'}

Use ONLY the provided reference documents to answer. If the documents don't cover something,
say so clearly. Always remind the student this is general guidance, not professional tax advice.

--- REFERENCE DOCUMENTS ---
{context}
--- END DOCUMENTS ---

Student's question: {question}

Provide a clear, helpful answer:"""

    # Try local LLaMA first
    local_answer, local_success = ask_ollama(prompt)
    if local_success:
        latency = round(time.time() - t0, 2)
        local_llm_counter.labels(status="success").inc()
        return local_answer, latency, estimate_tokens(prompt), estimate_tokens(local_answer), False, "llama3.2"

    # Route to Gemini
    local_llm_counter.labels(status="fallback").inc()
    answer, latency, inp, out, used_ext_fallback = ask_gemini(student_info, context, question, chunks)
    model = "extractive_fallback" if used_ext_fallback else "gemini-2.0-flash"
    return answer, latency, inp, out, used_ext_fallback, model


# --- Routes ---
@app.get("/")
def index():
    return FileResponse(os.path.join(BASE_DIR, "static", "index.html"))


@app.post("/chat")
def chat(req: ChatRequest):
    question = req.question.strip()
    student_info = req.student_info

    # Guard 1: keyword filter
    if not is_tax_question(question):
        refused_counter.labels(reason="not_tax_question").inc()
        return {
            "answer": "This doesn't appear to be a tax question. I can only help with U.S. tax questions for international students.",
            "confidence": 0,
            "refused": True,
            "reason": "not_tax_question",
        }

    query = build_query(student_info, question)
    t_ret = time.time()
    chunks, confidence = retriever.retrieve(query, top_k=TOP_K)
    retrieval_latency = round(time.time() - t_ret, 2)
    retrieval_latency_metric.observe(retrieval_latency)
    confidence_metric.observe(confidence)

    # Guard 2: confidence threshold
    if confidence < CONFIDENCE_THRESHOLD:
        refused_counter.labels(reason="low_confidence").inc()
        return {
            "answer": f"I couldn't find reliable information in my tax documents for that question (confidence: {confidence:.2f}). Try rephrasing, or consult a tax professional.",
            "confidence": round(confidence, 2),
            "refused": True,
            "reason": "low_confidence",
        }

    context = format_context(chunks)
    answer, llm_latency, input_tokens, output_tokens, used_fallback, model_used = ask_llm_with_routing(
        student_info, context, question, chunks
    )
    total_latency = round(retrieval_latency + llm_latency, 2)
    llm_latency_metric.observe(llm_latency)
    if used_fallback:
        fallback_counter.inc()

    # Log query
    query_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "confidence": round(confidence, 4),
        "retrieval_latency_s": retrieval_latency,
        "llm_latency_s": llm_latency,
        "total_latency_s": total_latency,
        "input_tokens_est": input_tokens,
        "output_tokens_est": output_tokens,
        "used_fallback": used_fallback,
    }
    if supabase:
        try:
            supabase.table("query_logs").insert(query_entry).execute()
        except Exception as e:
            print(f"Supabase query log error: {e}")
    else:
        with open(QUERY_LOG_PATH, 'a') as f:
            f.write(json.dumps(query_entry) + "\n")

    return {
        "answer": answer,
        "confidence": round(confidence, 2),
        "retrieval_latency": retrieval_latency,
        "llm_latency": llm_latency,
        "total_latency": total_latency,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "used_fallback": used_fallback,
        "model_used": model_used,
        "refused": False,
    }


@app.post("/feedback")
def feedback(req: FeedbackRequest):
    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": req.question,
        "answer_snippet": req.answer[:200],
        "rating": req.rating,
    }
    if supabase:
        try:
            supabase.table("feedback_logs").insert(entry).execute()
        except Exception as e:
            print(f"Supabase feedback log error: {e}")
    else:
        with open(FEEDBACK_LOG_PATH, 'a') as f:
            f.write(json.dumps(entry) + "\n")
    return {"status": "ok"}


@app.get("/logs/feedback")
def logs_feedback():
    if supabase:
        result = supabase.table("feedback_logs").select("*").order("timestamp", desc=True).execute()
        entries = result.data
    elif os.path.exists(FEEDBACK_LOG_PATH):
        entries = []
        with open(FEEDBACK_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    else:
        entries = []
    thumbs_up = sum(1 for e in entries if e.get("rating") == 1)
    thumbs_down = sum(1 for e in entries if e.get("rating") == 0)
    return {"total": len(entries), "thumbs_up": thumbs_up, "thumbs_down": thumbs_down, "entries": entries}


@app.get("/logs/queries")
def logs_queries():
    if supabase:
        result = supabase.table("query_logs").select("*").order("timestamp", desc=True).execute()
        entries = result.data
    elif os.path.exists(QUERY_LOG_PATH):
        entries = []
        with open(QUERY_LOG_PATH) as f:
            for line in f:
                line = line.strip()
                if line:
                    entries.append(json.loads(line))
    else:
        entries = []
    return {"total": len(entries), "entries": entries}
