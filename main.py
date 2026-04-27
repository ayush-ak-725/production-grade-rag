from rag_pipeline import RAGPipeline

if __name__ == "__main__":
    rag = RAGPipeline()

    while True:
        q = input("\n💬 Ask: ")
        print("\n🤖 Answer:\n", rag.query(q))