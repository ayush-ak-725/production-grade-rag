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

from config_loader import load_yaml


# ---------------- LOGGING SETUP ---------------- #
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# ---------------- LOAD DOCUMENTS ---------------- #
def load_documents(data_path: str):
    docs = []

    if not os.path.exists(data_path):
        logger.warning(f"Data path does not exist: {data_path}")
        return docs

    files = os.listdir(data_path)
    logger.info(f"Found {len(files)} files in {data_path}")

    for file in files:
        path = os.path.join(data_path, file)

        try:
            if file.endswith(".pdf"):
                logger.info(f"Loading PDF: {file}")
                loader = PyPDFLoader(path)
                loaded_docs = loader.load()
                docs.extend(loaded_docs)
                logger.info(f"Loaded {len(loaded_docs)} pages from {file}")

            elif file.endswith(".md"):
                logger.info(f"Loading Markdown: {file}")
                loader = TextLoader(path)
                loaded_docs = loader.load()
                docs.extend(loaded_docs)
                logger.info(f"Loaded {len(loaded_docs)} chunks from {file}")

            else:
                logger.debug(f"Skipping unsupported file: {file}")

        except Exception as e:
            logger.error(f"Failed to load {file}: {str(e)}")

    logger.info(f"Total loaded documents: {len(docs)}")
    return docs


# ---------------- LOAD WEBPAGES ---------------- #
def load_webpages(urls: List[str]):
    if not urls:
        return []

    logger.info(f"Loading {len(urls)} webpages")

    try:
        loader = WebBaseLoader(urls)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents from web")
        return docs

    except Exception as e:
        logger.error(f"Failed to load webpages: {str(e)}")
        return []


# ---------------- INGEST PIPELINE ---------------- #
def ingest(data_path="data", urls=None):
    logger.info("🚀 Starting ingestion pipeline")

    tool_config = load_yaml("tool.yaml")

    # Load docs
    docs = load_documents(data_path)

    if urls:
        web_docs = load_webpages(urls)
        docs.extend(web_docs)

    if not docs:
        logger.warning("No documents found. Exiting ingestion.")
        return

    # ---------------- CHUNKING ---------------- #
    logger.info("🔪 Starting document chunking")

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
        tokenizer=tokenizer,
        chunk_size=tool_config["chunking"]["chunk_size"],
        chunk_overlap=tool_config["chunking"]["chunk_overlap"]
    )

    chunks = splitter.split_documents(docs)

    logger.info(f"Generated {len(chunks)} chunks")

    # ---------------- METADATA ---------------- #
    logger.info("🧾 Adding metadata to chunks")

    for chunk in chunks:
        if "source" not in chunk.metadata:
            chunk.metadata["source"] = chunk.metadata.get("file_path", "unknown")

    # ---------------- EMBEDDINGS ---------------- #
    logger.info("🧠 Initializing embeddings model")

    try:
        embeddings = OllamaEmbeddings(
            model=tool_config["embedding"]["model"]
        )
    except Exception as e:
        logger.error(f"Embedding model init failed: {str(e)}")
        return

    # ---------------- VECTOR STORE ---------------- #
    logger.info("📦 Storing embeddings into Chroma DB")

    try:
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory="./chroma_db"
        )
    except Exception as e:
        logger.error(f"Failed to store embeddings: {str(e)}")
        return

    logger.info(f"✅ Ingestion completed successfully with {len(chunks)} chunks")


# ---------------- ENTRY POINT ---------------- #
if __name__ == "__main__":
    ingest(data_path="data")