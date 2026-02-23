"""
Tax Advisor Bot for International Students
Uses RAG (ChromaDB + sentence-transformers) + Google Gemini free API

Retrieval: Hybrid (vector + BM25) with Reciprocal Rank Fusion
Guardrail: Confidence threshold — refuses to answer if retrieval score is too low
Tracking: Latency (retrieval + LLM) and token estimates logged to query_log.jsonl
"""
import os
import json
import time
from datetime import datetime

from dotenv import load_dotenv
import chromadb
from google import genai
from retriever import HybridRetriever

load_dotenv()  # loads GEMINI_API_KEY from .env file

# --- Config ---
BASE_DIR = os.path.dirname(__file__)
CHROMA_DIR = os.path.join(BASE_DIR, 'tax_rag_data', 'data_work', 'chroma_db')
QUERY_LOG_PATH = os.path.join(BASE_DIR, 'query_log.jsonl')
FEEDBACK_LOG_PATH = os.path.join(BASE_DIR, 'feedback_log.jsonl')
COLLECTION_NAME = "tax_docs"
TOP_K = 5
CONFIDENCE_THRESHOLD = 0.70  # vector similarity floor for tax questions
RETRY_ATTEMPTS = 3           # max Gemini retries before falling back
RETRY_BASE_DELAY = 2         # seconds — doubles each attempt: 2s, 4s, 8s

# Must contain at least one of these for the question to be considered tax-related
TAX_KEYWORDS = {
    "tax", "taxes", "form", "irs", "filing", "file", "income", "deduction",
    "refund", "w-2", "w2", "1040", "8843", "8233", "1098", "withholding",
    "treaty", "visa", "f-1", "f1", "j-1", "j1", "opt", "cpt", "fica",
    "ssn", "itin", "scholarship", "stipend", "wage", "wages", "salary",
    "resident", "nonresident", "return", "credit", "exemption", "alien",
    "substantial", "presence", "deadline", "april", "extension", "state",
    "federal", "social security", "medicare", "fellowship", "tuition",
}


def estimate_tokens(text):
    """Rough token estimate: ~1 token per 4 characters (standard approximation)."""
    return len(text) // 4


def log_query(entry):
    """Append one query's stats as a JSON line to query_log.jsonl."""
    with open(QUERY_LOG_PATH, 'a') as f:
        f.write(json.dumps(entry) + "\n")


def collect_feedback(question, answer):
    """Ask the user to rate the answer and save to feedback_log.jsonl."""
    rating_input = input("Was this helpful? (y/n, or press Enter to skip): ").strip().lower()
    if rating_input not in ("y", "n"):
        return  # skipped

    rating = 1 if rating_input == "y" else 0
    label = "👍 Helpful" if rating == 1 else "👎 Not helpful"
    print(f"  Feedback recorded: {label}")

    entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "question": question,
        "answer_snippet": answer[:200],
        "rating": rating,
    }
    with open(FEEDBACK_LOG_PATH, 'a') as f:
        f.write(json.dumps(entry) + "\n")


def get_student_info():
    """Ask the student a few questions to personalize advice."""
    print("\n=== International Student Tax Advisor ===\n")
    print("I'll ask a few questions to give you the right tax advice.\n")

    visa_type = input("1. What is your visa type? (F-1 / J-1 / M-1 / Other): ").strip() or "F-1"
    home_country = input("2. What is your home country?: ").strip() or "India"
    first_year = input("3. What year did you first enter the U.S.?: ").strip() or "2023"
    tax_year = input("4. What tax year are you filing for? (e.g. 2024): ").strip() or "2024"

    print("5. What types of income did you have? (comma-separated)")
    print("   Options: Wages/Salary, Scholarship/Fellowship, Stipend, On-campus job, OPT/CPT, None")
    income_raw = input("   Your answer: ").strip() or "None"
    income_types = [x.strip() for x in income_raw.split(",")]

    state = input("6. What U.S. state do you live in?: ").strip() or "CA"
    has_ssn = input("7. Do you have an SSN or ITIN? (yes/no): ").strip().lower()
    has_ssn = has_ssn in ("yes", "y")

    return {
        "visa_type": visa_type,
        "home_country": home_country,
        "first_entry_year": first_year,
        "tax_year": tax_year,
        "income_types": income_types,
        "state": state,
        "has_ssn_or_itin": has_ssn,
    }


def is_tax_question(question):
    """Return True if the question contains at least one tax-related keyword."""
    q_lower = question.lower()
    return any(kw in q_lower for kw in TAX_KEYWORDS)


def build_query(student_info, question):
    """Enrich the query with student profile context for better retrieval."""
    parts = [
        question,
        f"{student_info['visa_type']} student",
        f"from {student_info['home_country']}",
        f"tax year {student_info['tax_year']}",
    ]
    if any("OPT" in t or "CPT" in t for t in student_info['income_types']):
        parts.append("OPT CPT employment")
    return " ".join(parts)


def format_context(chunks):
    """Format retrieved chunks into a readable context string."""
    parts = []
    for c in chunks:
        meta = c["metadata"]
        source = f"[{meta.get('title', 'Unknown')} - p.{meta.get('page_number', '?')}]"
        parts.append(f"{source}\n{c['text']}")
    return "\n\n---\n\n".join(parts)


def extractive_fallback(chunks):
    """
    Fallback when Gemini is unavailable.
    Returns the top 2 retrieved chunks as plain text so the user still gets useful info.
    """
    lines = ["[Gemini unavailable — showing raw source excerpts instead]\n"]
    for i, c in enumerate(chunks[:2], 1):
        meta = c["metadata"]
        source = f"{meta.get('title', 'Unknown')} (p.{meta.get('page_number', '?')})"
        lines.append(f"Source {i}: {source}\n{c['text'].strip()}")
    lines.append("\nNote: This is general guidance, not professional tax advice.")
    return "\n\n".join(lines)


def ask_gemini(student_info, context, question, chunks):
    """
    Send the question + context to Gemini with exponential backoff retry.

    Retry logic:
      - Attempt 1: try immediately
      - Attempt 2: wait 2s, retry
      - Attempt 3: wait 4s, retry
      - All failed: fall back to extractive_fallback() (no Gemini call)

    Returns (answer, latency_s, input_tokens, output_tokens, used_fallback).
    """
    prompt = f"""You are a helpful tax advisor for international students in the U.S.

Student profile:
- Visa: {student_info['visa_type']}
- Home country: {student_info['home_country']}
- First U.S. entry: {student_info['first_entry_year']}
- Tax year: {student_info['tax_year']}
- Income types: {', '.join(student_info['income_types'])}
- State: {student_info['state']}
- Has SSN/ITIN: {'Yes' if student_info['has_ssn_or_itin'] else 'No'}

Use ONLY the provided reference documents to answer. If the documents don't cover something,
say so clearly. Always remind the student this is general guidance, not professional tax advice.

--- REFERENCE DOCUMENTS ---
{context}
--- END DOCUMENTS ---

Student's question: {question}

Provide a clear, helpful answer:"""

    client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
    t0 = time.time()
    last_error = None

    for attempt in range(1, RETRY_ATTEMPTS + 1):
        if attempt > 1:
            delay = RETRY_BASE_DELAY * (2 ** (attempt - 2))  # 2s, 4s
            print(f"  [Retry {attempt}/{RETRY_ATTEMPTS} after {delay}s...]")
            time.sleep(delay)
        try:
            response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
            latency = round(time.time() - t0, 2)
            return response.text, latency, estimate_tokens(prompt), estimate_tokens(response.text), False
        except Exception as e:
            last_error = e
            print(f"  [Gemini error (attempt {attempt}/{RETRY_ATTEMPTS}): {e}]")

    # All retries exhausted — use extractive fallback
    latency = round(time.time() - t0, 2)
    answer = extractive_fallback(chunks)
    return answer, latency, 0, 0, True


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY in .env")
        print("Get a free key at: https://aistudio.google.com/apikey")
        return

    print("Connecting to ChromaDB...")
    client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = client.get_collection(name=COLLECTION_NAME)

    print("Building hybrid retriever (vector + BM25)...")
    retriever = HybridRetriever(collection)

    student_info = get_student_info()

    with open(os.path.join(BASE_DIR, "user_profile.json"), 'w') as f:
        json.dump(student_info, f, indent=2)

    print("\n=== Ready! Ask your tax questions (type 'quit' to exit) ===\n")

    while True:
        question = input("\nYour question: ").strip()
        if not question or question.lower() in ('quit', 'exit', 'q'):
            print("Goodbye! Remember: consult a tax professional for specific advice.")
            break

        # Refusal check 1: keyword filter — is this even a tax question?
        if not is_tax_question(question):
            print("\nThis question doesn't appear to be tax-related. I can only help with "
                  "U.S. tax questions for international students.\n")
            print("-" * 60)
            continue

        query = build_query(student_info, question)

        t_retrieval_start = time.time()
        print("Searching tax documents (hybrid)...")
        chunks, confidence = retriever.retrieve(query, top_k=TOP_K)
        retrieval_latency = round(time.time() - t_retrieval_start, 2)

        # Refusal check 2: confidence threshold — did we find relevant docs?
        if confidence < CONFIDENCE_THRESHOLD:
            print(f"\n[Low confidence: {confidence:.2f}] I couldn't find reliable information "
                  f"in my tax documents to answer that.\n"
                  f"Try rephrasing, or consult a tax professional or irs.gov.\n")
            print("-" * 60)
            continue

        context = format_context(chunks)
        print(f"[Confidence: {confidence:.2f}] Generating answer...\n")

        answer, llm_latency, input_tokens, output_tokens, used_fallback = ask_gemini(
            student_info, context, question, chunks
        )
        total_latency = round(retrieval_latency + llm_latency, 2)

        print(f"\n{answer}\n")
        mode = "fallback" if used_fallback else f"~{input_tokens} in / {output_tokens} out tokens"
        print(f"[Retrieval: {retrieval_latency}s | LLM: {llm_latency}s | Total: {total_latency}s | {mode}]")

        collect_feedback(question, answer)
        print("-" * 60)

        # Log this query's stats
        log_query({
            "timestamp": datetime.utcnow().isoformat(),
            "question": question,
            "confidence": round(confidence, 4),
            "retrieval_latency_s": retrieval_latency,
            "llm_latency_s": llm_latency,
            "total_latency_s": total_latency,
            "input_tokens_est": input_tokens,
            "output_tokens_est": output_tokens,
            "used_fallback": used_fallback,
        })


if __name__ == "__main__":
    main()
