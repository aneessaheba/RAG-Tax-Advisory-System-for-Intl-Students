"""
RAG Evaluation Script
Measures 5 metrics for each test question:
  1. Context Relevance  - how well retrieved chunks match the query (cosine similarity)
  2. Precision@K        - fraction of top-K chunks that contain at least one expected keyword
  3. Answer Relevance   - how well the generated answer addresses the question (cosine similarity)
  4. Faithfulness       - how grounded the answer is in the retrieved context (cosine similarity)
  5. LLM Judge Score    - Gemini scores the answer on correctness, completeness, groundedness
All metrics are 0.0 to 1.0 (higher is better).
"""
import os
import re
import json
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import chromadb
from google import genai
from retriever import HybridRetriever

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
CHROMA_DIR = os.path.join(BASE_DIR, 'tax_rag_data', 'data_work', 'chroma_db')
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, 'ground_truth.json')
RESULTS_PATH = os.path.join(BASE_DIR, 'evaluation_results.json')

EMBED_MODEL = "all-MiniLM-L6-v2"
COLLECTION_NAME = "tax_docs"
TOP_K = 5


def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def context_relevance(query_embedding, chunk_texts, embed_model):
    """Average cosine similarity between the query and each retrieved chunk."""
    chunk_embeddings = embed_model.encode(chunk_texts)
    scores = [cosine_similarity(query_embedding, ce) for ce in chunk_embeddings]
    return round(float(np.mean(scores)), 4)


def precision_at_k(chunk_texts, expected_keywords, k=None):
    """
    Fraction of top-K retrieved chunks that contain at least one expected keyword.
    Unlike binary hit rate, this measures how many of the K chunks are actually relevant.
    E.g. 4 out of 5 chunks contain keywords -> Precision@5 = 0.80
    """
    if k is None:
        k = len(chunk_texts)
    chunks_to_check = chunk_texts[:k]
    relevant = sum(
        1 for chunk in chunks_to_check
        if any(kw.lower() in chunk.lower() for kw in expected_keywords)
    )
    return round(relevant / k, 4)


def recall_at_k(chunk_texts, expected_keywords, k=None):
    """
    Fraction of expected keywords that appear in at least one of the top-K chunks.
    Unlike Precision@K (chunk-centric), Recall@K is keyword-centric:
    E.g. 3 of 4 expected keywords found anywhere in top-5 -> Recall@5 = 0.75
    """
    if not expected_keywords:
        return 0.0
    if k is None:
        k = len(chunk_texts)
    chunks_to_check = chunk_texts[:k]
    found = sum(
        1 for kw in expected_keywords
        if any(kw.lower() in chunk.lower() for chunk in chunks_to_check)
    )
    return round(found / len(expected_keywords), 4)


def answer_relevance(question_embedding, answer_embedding):
    """Cosine similarity between the question and the generated answer."""
    return round(cosine_similarity(question_embedding, answer_embedding), 4)


def faithfulness(answer_embedding, chunk_texts, embed_model):
    """Cosine similarity between the answer and the average context embedding."""
    chunk_embeddings = embed_model.encode(chunk_texts)
    avg_context = np.mean(chunk_embeddings, axis=0)
    return round(cosine_similarity(answer_embedding, avg_context), 4)


def llm_judge(question, context, answer, expected_note, api_key, model_name="gemini-2.0-flash"):
    """
    Use Gemini to score the answer on 3 dimensions (0.0 to 1.0 each):
      - correctness:   Is the answer factually accurate for U.S. tax law?
      - completeness:  Does it fully address what the question asked?
      - groundedness:  Is it based on the provided context (not hallucinated)?

    Returns a dict with the 3 scores and an overall average.
    Falls back to 0.0 scores if parsing fails.
    """
    prompt = f"""You are evaluating a RAG system that answers U.S. tax questions for international students.

Score the ANSWER below on 3 dimensions. Each score must be a number from 0.0 to 1.0.

Question: {question}
Expected fact: {expected_note}

Context provided to the system:
{context[:3000]}

Generated answer:
{answer}

Score these 3 dimensions:
1. correctness   - Is the answer factually correct based on U.S. tax law and the expected fact?
2. completeness  - Does the answer fully address the question (not missing key points)?
3. groundedness  - Is the answer based on the provided context (not making things up)?

Reply ONLY with valid JSON in exactly this format (no extra text):
{{"correctness": 0.0, "completeness": 0.0, "groundedness": 0.0}}"""

    try:
        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(model=model_name, contents=prompt)
        raw = response.text.strip()

        # Extract JSON even if model adds extra text
        match = re.search(r'\{[^}]+\}', raw)
        if not match:
            raise ValueError(f"No JSON found in response: {raw}")

        scores = json.loads(match.group())
        correctness = round(float(scores.get("correctness", 0.0)), 4)
        completeness = round(float(scores.get("completeness", 0.0)), 4)
        groundedness = round(float(scores.get("groundedness", 0.0)), 4)
        overall = round((correctness + completeness + groundedness) / 3, 4)

        return {
            "correctness": correctness,
            "completeness": completeness,
            "groundedness": groundedness,
            "overall": overall,
        }
    except Exception as e:
        print(f"    [LLM Judge error: {e}]")
        return {"correctness": 0.0, "completeness": 0.0, "groundedness": 0.0, "overall": 0.0}


def generate_answer(question, chunk_texts, api_key, model_name="gemini-2.0-flash"):
    """Call Gemini to generate an answer grounded in the retrieved chunks."""
    context = "\n\n".join(chunk_texts)
    prompt = f"""You are a tax advisor for international students.
Answer the question using ONLY the context below. Be concise.

Context:
{context}

Question: {question}
Answer:"""
    client = genai.Client(api_key=api_key)
    response = client.models.generate_content(model=model_name, contents=prompt)
    return response.text.strip()


def main():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY in .env")
        return

    print("Loading embedding model...")
    embed_model = SentenceTransformer(EMBED_MODEL)

    print("Connecting to ChromaDB...")
    db_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = db_client.get_collection(name=COLLECTION_NAME)

    print("Building hybrid retriever...")
    retriever = HybridRetriever(collection)

    with open(GROUND_TRUTH_PATH, 'r') as f:
        test_cases = json.load(f)

    results = []
    totals = {
        "context_relevance": 0, "precision_at_k": 0, "recall_at_k": 0,
        "answer_relevance": 0, "faithfulness": 0,
        "llm_judge": 0,
    }

    print(f"\nRunning evaluation on {len(test_cases)} questions...\n")
    print(f"{'#':<4} {'Question':<50} {'CtxRel':>7} {'P@5':>6} {'R@5':>6} {'AnsRel':>7} {'Faith':>7} {'Judge':>7}")
    print("-" * 106)

    for i, tc in enumerate(test_cases):
        question = tc["question"]
        expected_keywords = tc["expected_keywords"]
        expected_note = tc.get("note", "")

        q_embedding = embed_model.encode(question).tolist()

        # Hybrid retrieval
        chunks, _ = retriever.retrieve(question, top_k=TOP_K)
        chunk_texts = [c["text"] for c in chunks]
        context_str = "\n\n".join(chunk_texts)

        # Generate answer
        answer = generate_answer(question, chunk_texts, api_key)
        answer_embedding = embed_model.encode(answer).tolist()

        # Compute cosine-based metrics
        m_context_relevance = context_relevance(q_embedding, chunk_texts, embed_model)
        m_precision = precision_at_k(chunk_texts, expected_keywords, k=TOP_K)
        m_recall = recall_at_k(chunk_texts, expected_keywords, k=TOP_K)
        m_answer_relevance = answer_relevance(q_embedding, answer_embedding)
        m_faithfulness = faithfulness(answer_embedding, chunk_texts, embed_model)

        # LLM-as-a-Judge
        judge_scores = llm_judge(question, context_str, answer, expected_note, api_key)
        m_judge = judge_scores["overall"]

        result = {
            "question": question,
            "expected_keywords": expected_keywords,
            "answer": answer,
            "metrics": {
                "context_relevance": m_context_relevance,
                "precision_at_k": m_precision,
                "recall_at_k": m_recall,
                "answer_relevance": m_answer_relevance,
                "faithfulness": m_faithfulness,
                "llm_judge": m_judge,
                "llm_judge_breakdown": judge_scores,
            }
        }
        results.append(result)

        for k in totals:
            totals[k] += result["metrics"][k]

        short_q = question[:48] + ".." if len(question) > 48 else question
        print(f"{i+1:<4} {short_q:<50} {m_context_relevance:>7.3f} {m_precision:>6.2f} {m_recall:>6.2f} {m_answer_relevance:>7.3f} {m_faithfulness:>7.3f} {m_judge:>7.3f}")

    n = len(test_cases)
    averages = {k: round(v / n, 4) for k, v in totals.items()}

    print("-" * 106)
    print(f"{'AVERAGE':<54} {averages['context_relevance']:>7.3f} {averages['precision_at_k']:>6.2f} {averages['recall_at_k']:>6.2f} {averages['answer_relevance']:>7.3f} {averages['faithfulness']:>7.3f} {averages['llm_judge']:>7.3f}")

    output = {
        "retrieval": "hybrid (vector + BM25 + RRF)",
        "model": EMBED_MODEL,
        "llm": "gemini-2.0-flash",
        "top_k": TOP_K,
        "num_questions": n,
        "averages": averages,
        "per_question": results,
    }
    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")
    print("\nSummary:")
    print(f"  Context Relevance : {averages['context_relevance']:.3f}  (are retrieved chunks related to the question?)")
    print(f"  Precision@{TOP_K}       : {averages['precision_at_k']:.2f}   (fraction of top-{TOP_K} chunks that contain expected keywords)")
    print(f"  Recall@{TOP_K}          : {averages['recall_at_k']:.2f}   (fraction of expected keywords found in top-{TOP_K} chunks)")
    print(f"  Answer Relevance  : {averages['answer_relevance']:.3f}  (does the answer address the question?)")
    print(f"  Faithfulness      : {averages['faithfulness']:.3f}  (is the answer grounded in retrieved context?)")
    print(f"  LLM Judge Score   : {averages['llm_judge']:.3f}  (Gemini rates correctness + completeness + groundedness)")


if __name__ == "__main__":
    main()
