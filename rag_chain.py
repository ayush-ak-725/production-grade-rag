from langchain_chroma import Chroma
from langchain_ollama import OllamaLLM, OllamaEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from config_loader import load_yaml


def build_rag_chain():
    agent_config = load_yaml("agent.yaml")
    tool_config = load_yaml("tool.yaml")

    # LLM
    llm = OllamaLLM(
        model=agent_config["llm"]["model"],
        temperature=agent_config["llm"]["temperature"]
    )

    # Embeddings
    embeddings = OllamaEmbeddings(
        model=tool_config["embedding"]["model"]
    )

    # Vector store
    vectorstore = Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": agent_config["retrieval"]["k"]}
    )

    # Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", agent_config["prompt"]["system"]),
        ("human", agent_config["prompt"]["user"])
    ])

    return llm, retriever, prompt


def format_docs(docs):
    context = ""
    for doc in docs:
        context += f"{doc.page_content}\n[source: {doc.metadata.get('source')}]\n\n"
    return context


def query_rag(question: str):
    llm, retriever, prompt = build_rag_chain()

    docs = retriever.invoke(question)

    context = format_docs(docs)

    chain = prompt | llm

    response = chain.invoke({
        "context": context,
        "question": question
    })

    return response