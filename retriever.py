from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_community.retrievers import BM25Retriever
from config_loader import load_yaml


class HybridRetriever:
    def __init__(self, docs):
        tool_config = load_yaml("tool.yaml")

        self.embeddings = OllamaEmbeddings(
            model=tool_config["embedding"]["model"]
        )

        self.vectorstore = Chroma(
            persist_directory="./chroma_db",
            embedding_function=self.embeddings
        )

        self.vector_retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": 6}
        )

        self.bm25_retriever = BM25Retriever.from_documents(docs)
        self.bm25_retriever.k = 6

    def retrieve(self, query):
        vector_docs = self.vector_retriever.invoke(query)
        bm25_docs = self.bm25_retriever.invoke(query)

        # Merge + deduplicate
        seen = set()
        combined = []

        for doc in vector_docs + bm25_docs:
            key = doc.page_content[:100]
            if key not in seen:
                seen.add(key)
                combined.append(doc)

        return combined