"""
RAGAS Evaluation — uses industry-standard RAGAS framework with Gemini.

Metrics computed:
  - faithfulness       : are all claims in the answer supported by the context?
  - answer_relevancy   : how relevant is the answer to the question?
  - context_precision  : is the retrieved context relevant to the question?
  - context_recall     : does the context cover the ground truth answer?

Run with:
    python ragas_evaluate.py
"""
import os
import json
from dotenv import load_dotenv
import chromadb
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import HuggingFaceEmbeddings
from google import genai as google_genai
from retriever import HybridRetriever

load_dotenv()

BASE_DIR = os.path.dirname(__file__)
CHROMA_DIR = os.path.join(BASE_DIR, 'tax_rag_data', 'data_work', 'chroma_db')
GROUND_TRUTH_PATH = os.path.join(BASE_DIR, 'ground_truth.json')
RESULTS_PATH = os.path.join(BASE_DIR, 'ragas_results.json')
COLLECTION_NAME = "tax_docs"
TOP_K = 5


def generate_answer(question, chunk_texts, api_key):
    """Generate answer using Gemini for evaluation."""
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
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        print("ERROR: Set GEMINI_API_KEY in .env")
        return

    print("Connecting to ChromaDB...")
    db_client = chromadb.PersistentClient(path=CHROMA_DIR)
    collection = db_client.get_collection(name=COLLECTION_NAME)

    print("Building hybrid retriever...")
    retriever = HybridRetriever(collection)

    with open(GROUND_TRUTH_PATH, 'r') as f:
        test_cases = json.load(f)

    # Configure RAGAS to use Gemini
    print("Configuring RAGAS with Gemini...")
    llm = LangchainLLMWrapper(ChatGoogleGenerativeAI(
        model="gemini-2.0-flash",
        google_api_key=api_key,
        temperature=0,
    ))
    embeddings = HuggingFaceEmbeddings(model="all-MiniLM-L6-v2")

    # Build RAGAS dataset
    print(f"\nBuilding evaluation dataset from {len(test_cases)} questions...")
    questions, answers, contexts_list, ground_truths = [], [], [], []

    for i, tc in enumerate(test_cases):
        question = tc["question"]
        ground_truth = tc.get("note", "")

        chunks, _ = retriever.retrieve(question, top_k=TOP_K)
        chunk_texts = [c["text"] for c in chunks]

        answer = generate_answer(question, chunk_texts, api_key)

        questions.append(question)
        answers.append(answer)
        contexts_list.append(chunk_texts)
        ground_truths.append(ground_truth)

        print(f"  [{i+1}/{len(test_cases)}] {question[:60]}...")

    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths,
    })

    # Run RAGAS evaluation
    print("\nRunning RAGAS evaluation (this may take a few minutes)...")
    results = evaluate(
        dataset=dataset,
        metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
        llm=llm,
        embeddings=embeddings,
    )

    print("\n" + "="*60)
    print("  RAGAS Evaluation Results")
    print("="*60)
    df = results.to_pandas()
    results_dict = df.select_dtypes(include='number').mean().to_dict()
    for metric, score in results_dict.items():
        print(f"  {metric:<25}: {score:.4f}")
    print("="*60)

    output = {
        "framework": "RAGAS",
        "llm": "gemini-2.0-flash",
        "embeddings": "all-MiniLM-L6-v2 (HuggingFace)",
        "retrieval": "hybrid (vector + BM25 + RRF)",
        "top_k": TOP_K,
        "num_questions": len(test_cases),
        "averages": {k: round(float(v), 4) for k, v in results_dict.items()},
        "per_question": df.to_dict(orient="records"),
    }

    with open(RESULTS_PATH, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
