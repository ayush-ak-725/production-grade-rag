import os
from datetime import datetime

from rag_pipeline import RAGPipeline   # your class file
from reranker import Reranker  # your class file


OUTPUT_FILE = "test/ragas_dataset.json"


def load_existing_data():
    if not os.path.exists(OUTPUT_FILE):
        return []

    with open(OUTPUT_FILE, "r") as f:
        return json.load(f)


def save_data(data):
    with open(OUTPUT_FILE, "w") as f:
        json.dump(data, f, indent=2)


def run_queries(test_cases):
    rag = RAGPipeline()
    dataset = load_existing_data()

    for i, tc in enumerate(test_cases, 1):
        question = tc["question"]
        ground_truth = tc["ground_truth"]

        print(f"\n🔹 Running {i}: {question}")

        # Run pipeline
        answer = rag.query(question)

        # 🔥 IMPORTANT: get retrieved docs separately
        docs = rag.retriever.retrieve(question)
        reranker = Reranker()
        docs, scores = reranker.rerank(question, docs, 4, return_scores=True)
        contexts = [doc.page_content for doc in docs]

        entry = {
            "question": question,
            "answer": answer,
            "contexts": contexts,
            "ground_truth": ground_truth,
            "timestamp": datetime.utcnow().isoformat()
        }

        dataset.append(entry)
        save_data(dataset)

        print("✅ Saved")

import json

def load_test_cases(file_path: str):
    try:
        with open(file_path, "r") as f:
            data = json.load(f)

        # basic validation
        for item in data:
            if "question" not in item or "ground_truth" not in item:
                raise ValueError("Invalid format in JSON file")

        return data

    except Exception as e:
        print(f"❌ Failed to load test cases: {e}")
        return []

if __name__ == "__main__":
    test_cases = load_test_cases("test/question_ground_truth.json")

    if not test_cases:
        print("No test cases found. Exiting.")
    else:
        run_queries(test_cases)