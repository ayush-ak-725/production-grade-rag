from dotenv import load_dotenv
load_dotenv()

from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langfuse import Langfuse

from retriever import HybridRetriever
from reranker import Reranker
from evaluation import extract_claims, score_claims, compute_coverage
from config_loader import load_yaml
from ingest import load_documents

# Initialize Langfuse
langfuse = Langfuse()


class RAGPipeline:
    def __init__(self):
        agent_config = load_yaml("agent.yaml")
        prompt_config = load_yaml("prompts.yaml")

        self.llm = OllamaLLM(
            model=agent_config["llm"]["model"],
            temperature=0.0
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_config["rag_prompt"]["system"]),
            ("human", prompt_config["rag_prompt"]["user"])
        ])

        # Load docs (for BM25)
        docs = load_documents("data")

        self.retriever = HybridRetriever(docs)
        self.reranker = Reranker()

    def format_context(self, docs):
        context = ""
        for doc in docs:
            context += f"{doc.page_content}\n[source: {doc.metadata.get('source')}]\n\n"
        return context

    def is_answerable(self, docs):
        return len(docs) > 0

    def query(self, question: str):
        trace = langfuse.trace(
            name="rag_pipeline",
            input={"question": question}
        )

        # -------------------------
        # Step 1: Retrieval
        # -------------------------
        span = trace.span(name="retrieval")

        docs = self.retriever.retrieve(question)

        span.update(output={
            "num_docs": len(docs),
            "docs": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source")
                }
                for doc in docs
            ]
        })
        span.end()

        # -------------------------
        # Step 2: Rerank
        # -------------------------
        top_k = load_yaml("tool.yaml")['tools']['reranker']['top_k']

        span = trace.span(name="reranking")

        docs, scores = self.reranker.rerank(
            question, docs, top_k, return_scores=True
        )

        span.update(output={
            "scores": scores.tolist(),
            "docs": [
                {
                    "content": doc.page_content,
                    "source": doc.metadata.get("source")
                }
                for doc in docs
            ]
        })
        span.end()

        # -------------------------
        # Step 3: Grounding check
        # -------------------------
        if not self.is_answerable(docs):
            trace.update(output={"answer": "Not answerable"})
            return "I don't have enough information to answer that."

        context = self.format_context(docs)

        # -------------------------
        # Step 4: Prompt
        # -------------------------
        span = trace.span(name="prompt")

        span.update(output={
            "context": context,
            "question": question
        })
        span.end()

        chain = self.prompt | self.llm

        # -------------------------
        # Step 5: LLM
        # -------------------------
        span = trace.span(name="llm_call")

        response = chain.invoke({
            "context": context,
            "question": question
        })

        span.update(output={"response": response})
        span.end()

        # -------------------------
        # Step 6: Evaluation (NEW 🔥)
        # -------------------------
        eval_span = trace.span(name="evaluation")

        claims = extract_claims(response)
        claim_scores = score_claims(claims, docs, self.reranker)

        coverage_metrics = compute_coverage(claim_scores)

        # Faithfulness proxy (max support score across claims)
        faithfulness_score = max(
            [c["max_score"] for c in claim_scores],
            default=-999
        )

        print("\n" + "="*50)
        print("📊 RAG EVALUATION")
        print("="*50)

        print(f"\n✅ Coverage: {coverage_metrics['coverage']:.2f} "
              f"({coverage_metrics['supported']}/{coverage_metrics['total']})")

        print(f"📈 Faithfulness Score: {faithfulness_score:.3f}")

        print("\n🧠 Claims:")
        for i, c in enumerate(claims, 1):
            print(f"{i}. {c}")

        print("\n🔍 Claim Support Scores:")
        for i, cs in enumerate(claim_scores, 1):
            print(f"{i}. {cs['max_score']:.3f} → {cs['claim']}")

        print("="*50 + "\n")

        eval_span.update(output={
            "claims": claims,
            "claim_scores": claim_scores,
            "coverage": coverage_metrics,
            "faithfulness_score": faithfulness_score
        })
        eval_span.end()


# -------------------------
        # Final trace
        # -------------------------
        trace.update(
            output={"final_answer": response},
            metadata={
                "top_k": top_k,
                "model": self.llm.model
            }
        )

        return response