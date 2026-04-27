from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate

from retriever import HybridRetriever
from reranker import Reranker
from config_loader import load_yaml
from ingest import load_documents


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
        # 🔥 Simple grounding check
        return len(docs) > 0

    def query(self, question: str):
        # Step 1: Hybrid retrieval
        docs = self.retriever.retrieve(question)

        # Step 2: Rerank
        docs = self.reranker.rerank(question, docs)

        # Step 3: Citation enforcement
        if not self.is_answerable(docs):
            return "I don't have enough information to answer that."

        context = self.format_context(docs)

        chain = self.prompt | self.llm

        response = chain.invoke({
            "context": context,
            "question": question
        })

        return response