import json
from datasets import Dataset
from ragas import evaluate
from ragas.llms.base import LangchainLLMWrapper
from ragas.run_config import RunConfig
from ragas.metrics import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

# -----------------------------
# Run config (avoid timeouts)
# -----------------------------
run_config = RunConfig(
    max_workers=1,
    timeout=300
)

# -----------------------------
# 1. Load dataset
# -----------------------------
with open("test/ragas_dataset.json", "r") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# -----------------------------
# 2. Setup Ollama via OpenAI-compatible API
# -----------------------------
evaluator_llm = ChatOllama(
    model="llama3.1:8b",
    temperature=0,
    num_ctx=8192, # <--- FORCE 8k context here
)
eval_llm = LangchainLLMWrapper(evaluator_llm)

eval_embeddings = HuggingFaceEmbeddings(
    model="sentence-transformers/all-MiniLM-L6-v2"
)

# -----------------------------
# 4. Evaluate
# -----------------------------
result = evaluate(
    dataset=dataset,
    metrics=[Faithfulness(), AnswerRelevancy(), ContextPrecision(), ContextRecall()],  # ✅ MUST be in a list
    llm=eval_llm,             # ✅ Pass your local Ollama LLM
    embeddings=eval_embeddings, # ✅ Pass your local embeddings
    run_config=run_config
)

# Convert the result object to a Pandas DataFrame
df = result.to_pandas()

# Export to CSV for detailed inspection
df.to_csv("test/ragas_results.csv", index=False)

print("Results saved to ragas_results.csv. Open this file to see row-by-row scores!")


print(result)