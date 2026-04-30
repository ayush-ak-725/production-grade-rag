import os
import logging
from typing import List

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
    TextLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from transformers import AutoTokenizer

from config_loader import load_yaml
import os
os.environ["USER_AGENT"] = "MyResearchBot/1.0"


# ---------------- LOGGING SETUP ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)

# ---------------- HELPER: LOAD URLS FROM TXT ---------------- #
def get_urls_from_file(file_path: str) -> List[str]:
    """Reads a text file and returns a list of valid URLs."""
    if not os.path.exists(file_path):
        logger.warning(f"URL file not found: {file_path}")
        return []

    with open(file_path, "r") as f:
        # Strip whitespace and filter out empty lines or comments
        urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

    logger.info(f"Extracted {len(urls)} URLs from {file_path}")
    return urls

# ---------------- LOAD DOCUMENTS (Recursive for Multiple PDFs) ---------------- #
def load_documents(data_path: str):
    docs = []
    if not os.path.exists(data_path):
        return docs

    # Iterating through all files in the directory
    for root, _, files in os.walk(data_path):
        for file in files:
            path = os.path.join(root, file)
            try:
                if file.endswith(".pdf"):
                    # Use PyPDFLoader for research papers
                    loader = PyPDFLoader(path)
                    docs.extend(loader.load())
                elif file.endswith(".md") or file.endswith(".txt"):
                    loader = TextLoader(path)
                    docs.extend(loader.load())
            except Exception as e:
                logger.error(f"Error loading {file}: {e}")

    logger.info(f"Total pages/docs loaded from disk: {len(docs)}")
    return docs

# ---------------- INGEST PIPELINE ---------------- #
def ingest(data_path="data", url_file="data/web_pages.txt"):
    logger.info("🚀 Starting multi-source ingestion pipeline")
    tool_config = load_yaml("tool.yaml")

    # 1. Gather all documents
    all_docs = load_documents(data_path)

    # 2. Gather all web pages
    urls = get_urls_from_file(url_file)
    if urls:
        try:
            web_loader = WebBaseLoader(urls)
            web_docs = web_loader.load()
            all_docs.extend(web_docs)
            logger.info(f"Successfully integrated {len(web_docs)} web pages.")
        except Exception as e:
            logger.error(f"Web loading failed: {e}")

    if not all_docs:
        logger.error("No content found to ingest. Check your data/ folder and web_pages.txt.")
        return

    # 3. Enhanced Chunking for Research Papers
    # Research papers benefit from smaller chunks with higher overlap for context retention
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B")
    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=tool_config["chunking"]["chunk_size"],
        chunk_overlap=tool_config["chunking"]["chunk_overlap"]
    )
    chunks = splitter.split_documents(all_docs)
    logger.info(f"Generated {len(chunks)} total chunks.")

    # 4. Embeddings & Vector Store
    embeddings = OllamaEmbeddings(model=tool_config["embedding"]["model"])

    # We use .from_documents to create/update the store
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )

    logger.info("✅ Ingestion complete. Vector DB is ready for retrieval.")
    return vectorstore

if __name__ == "__main__":
    # Ensure the 'data' directory exists for your PDFs
    if not os.path.exists("data"):
        os.makedirs("data")

    ingest()