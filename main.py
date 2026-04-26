from rag_chain import query_rag

if __name__ == "__main__":
    while True:
        query = input("\n💬 Ask your question: ")
        answer = query_rag(query)
        print("\n🤖 Answer:\n", answer)