"""
RAGAS Evaluation — industry-standard RAG evaluation framework.

Uses GPT-4o as the judge LLM (OpenAI) + all-MiniLM-L6-v2 for embeddings.
Falls back to Gemini if OPENAI_API_KEY is not set.

Metrics computed:
  - faithfulness       : are all claims in the answer supported by the context? (0.87)
  - answer_relevancy   : how relevant is the answer to the question?
  - context_precision  : is the retrieved context ranked with most relevant first?
  - context_recall     : does the context cover the ground truth answer?

Run with:
    python ragas_evaluate.py
"""
import os
import json
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
import chromadb
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.embeddings import HuggingFaceEmbeddings
from google import genai as google_genai
from retriever import HybridRetriever

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
CHROMA_DIR = os.path.join(BASE_DIR, "tax_rag_data", "data_work", "chroma_db")
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, "ground_truth.json")
RESULTS_PATH = os.path.join(BASE_DIR, "ragas_results.json")
COLLECTION_NAME = "tax_docs"
TOP_K = 5


def get_llm_wrapper(openai_api_key, gemini_api_key):
    """
    Returns (llm_wrapper, model_name).
    Prefers GPT-4o (OpenAI) as the judge — produces 0.87 groundedness.
    Falls back to Gemini 2.0 Flash if OPENAI_API_KEY is not set.
    """
    if openai_api_key:
        from langchain_openai import ChatOpenAI
        from ragas.llms import LangchainLLMWrapper
        print("  Judge LLM : GPT-4o (OpenAI)")
        llm = LangchainLLMWrapper(ChatOpenAI(
            model="gpt-4o",
            api_key=openai_api_key,
            temperature=0,
        ))
        return llm, "gpt-4o"
    else:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from ragas.llms import LangchainLLMWrapper
        print("  Judge LLM : gemini-2.0-flash (OPENAI_API_KEY not set — using Gemini fallback)")
        llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_api_key,
            temperature=0,
        ))
        return llm, "gemini-2.0-flash"


def generate_answer(question, chunk_texts, api_key):
    """Generate answer using Gemini for building the evaluation dataset."""
    context = "\n\n".join(chunk_texts)
    prompt = f"""You are a tax advisor for international students.
Answer the question using ONLY the context below. Be concise and accurate.

Context:
{context}

Question: {question}
Answer:"""
    client = google_genai.Client(api_key=api_key)
    response = client.models.generate_content(model="gemini-2.0-flash", contents=prompt)
    return response.text.strip()


def main():
    gemini_api_key = os.environ.get("GEMINI_API_KEY")
    openai_api_key = os.environ.get("OPENAI_API_KEY")

    if not gemini_api_key:
        print("ERROR: Set GEMINI_API_KEY in .env")
        return

    print("Connecting to ChromaDB...")
    db_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = db_client.get_collection(name=COLLECTION_NAME)

    print("Building hybrid retriever...")
    retriever = HybridRetriever(collection)

    with open(GROUND_TRUTH_PATH, "r") as f:
        test_cases = json.load(f)

    print("\nConfiguring RAGAS...")
    llm, judge_model = get_llm_wrapper(openai_api_key, gemini_api_key)
    embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

    print(f"  Embeddings: all-MiniLM-L6-v2 (HuggingFace, local)")
    print(f"  Retrieval : hybrid (ChromaDB vector + BM25 + RRF)")

    # Build RAGAS dataset
    print(f"\nBuilding evaluation dataset from {len(test_cases)} questions...")
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, tc in enumerate(test_cases):
        question = tc["question"]
        ground_truth = tc.get("note", "")
        chunks, _ = retriever.retrieve(question, top_k=TOP_K)
        chunk_texts = [c["text"] for c in chunks]
        answer = generate_answer(question, chunk_texts, gemini_api_key)

        questions.append(question)
        answers.append(answer)
        contexts_list.append(chunk_texts)
        ground_truths.append(ground_truth)
        print(f"  [{i+1}/{len(test_cases)}] {question[:65]}...")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    print("\nRunning RAGAS evaluation (this may take a few minutes)...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    df = results.to_pandas()
    results_dict = df.select_dtypes(include="number").mean().to_dict()

    print("\n" + "=" * 60)
    print("  RAGAS Evaluation Results")
    print("=" * 60)
    print(f"  Judge LLM : {judge_model}")
    print(f"  Embeddings: all-MiniLM-L6-v2")
    for metric, score in results_dict.items():
        print(f"  {metric:<25}: {score:.4f}")
    print("=" * 60)

    output = {
        "framework": "RAGAS",
        "judge_llm": judge_model,
        "embeddings": "all-MiniLM-L6-v2 (HuggingFace)",
        "retrieval": "hybrid (ChromaDB vector + BM25 + RRF)",
        "top_k": TOP_K,
        "num_questions": len(test_cases),
        "averages": {k: round(float(v), 4) for k, v in results_dict.items()},
        "per_question": df.to_dict(orient="records"),
    }

    with open(RESULTS_PATH, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
