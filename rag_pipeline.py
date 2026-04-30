import time

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
from llmlingua import PromptCompressor
from sarvamai import SarvamAI
import tiktoken
import re

# Initialize Langfuse
langfuse = Langfuse()


class RAGPipeline:
    def __init__(self):
        agent_config = load_yaml("agent.yaml")
        prompt_config = load_yaml("prompts.yaml")
        self.compressor = PromptCompressor(
            model_name="microsoft/llmlingua-2-bert-base-multilingual-cased-meetingbank",
            use_llmlingua2=True,
            device_map="mps"
        )
        self.llm = OllamaLLM(
            model=agent_config["llm"]["model"],
            temperature=0.0,
            num_ctx=8192,
        )

        self.prompt = ChatPromptTemplate.from_messages([
            ("system", prompt_config["rag_prompt"]["system"]),
            ("human", prompt_config["rag_prompt"]["user"])
        ])

        # Load docs (for BM25)
        docs = load_documents("data")

        self.retriever = HybridRetriever(docs)
        self.reranker = Reranker()
        # 1. Initialize the tokenizer here
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

        self.client = SarvamAI(
            api_subscription_key="sk_nz0ueg9r_dNXbKDtPJ1NJCtpOruyJsY4j",
        )

    def count_tokens(self, text: str) -> int:
        return len(self.tokenizer.encode(text))

    def remove_redundancy(self, docs, threshold=0.85):
        """Lexical redundancy check with detailed logging."""
        unique_docs = []
        seen_contents = []

        print(f"\n🔄  REDUNDANCY CHECK (Threshold: {threshold})")
        initial_count = len(docs)

        for i, doc in enumerate(docs):
            is_redundant = False
            content_a = doc.page_content.lower()
            set_a = set(content_a.split())

            for j, seen in enumerate(seen_contents):
                set_b = set(seen.split())
                intersection = set_a.intersection(set_b)
                overlap = len(intersection) / max(len(set_a), len(set_b))

                if overlap > threshold:
                    is_redundant = True
                    source_a = doc.metadata.get('source', 'Unknown')
                    print(f"   ❌ Dropping doc {i} (from {source_a}) - {overlap*100:.1f}% overlap with unique doc {j}")
                    break

            if not is_redundant:
                unique_docs.append(doc)
                seen_contents.append(content_a)

        removed = initial_count - len(unique_docs)
        print(f"✅  Redundancy Filter Complete: {len(unique_docs)} unique docs kept, {removed} removed.")
        return unique_docs

    def compress_with_lingua(self, question, context):
        """Uses LLMLingua-2 to remove low-information tokens."""
        # LLMLingua-2 uses a much simpler argument set
        compressed_prompt = self.compressor.compress_prompt(
            context,
            question=question,
            rate=0.4,              # Target compression rate (e.g., keep 40% of tokens)
            force_tokens=['\n'],   # Ensure newlines aren't stripped to keep source citations clear
            drop_consecutive=True
        )
        return compressed_prompt["compressed_prompt"]

    def format_context(self, docs):
        context = ""
        for doc in docs:
            context += f"{doc.page_content}\n[source: {doc.metadata.get('source')}]\n\n"
        return context

    def is_answerable(self, docs):
        return len(docs) > 0

    def speak_response(self, text, output_file="tts/voice_response/response.wav"):
        """
        Cleans RAG text for natural speech and converts to voice.
        """
        # 1. Remove the "Citations" / "References" block at the end
        # This splits the text at the word 'Citations' and takes only the first part
        clean_text = re.split(r'\n\**Citations\**:', text, flags=re.IGNORECASE)[0]
        clean_text = re.split(r'\n\**References\**:', clean_text, flags=re.IGNORECASE)[0]

        # 2. Remove inline citations like (Author, 2021) or (Author et al., 2018)
        # Pattern: Look for '(' followed by text and a 4-digit year, ending with ')'
        clean_text = re.sub(r'\([^)]*\d{4}[^)]*\)', '', clean_text)

        # 3. Final Polish: Remove markdown bolding (**), bullet points (*), and extra whitespace
        clean_text = clean_text.replace("**", "").replace("*", "")
        clean_text = re.sub(r'\s+', ' ', clean_text).strip()

        print(f"🎙️ Cleaned for Voice: {clean_text[:100]}...")

        # --- Sarvam Streaming Logic ---
        chunks = []
        for chunk in self.client.text_to_speech.convert_stream(
                text=clean_text,
                target_language_code="en-IN",
                model="bulbul:v3",
                speaker="shubh"
        ):
            chunks.append(chunk)

        audio = b"".join(chunks)
        with open(output_file, "wb") as f:
            f.write(audio)

        return output_file

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

        # ... (Step 1: Retrieval & Step 2: Rerank) ...

        # -------------------------
        # Step 2.5: Context Compression (NEW 🔥)
        # -------------------------
        comp_span = trace.span(name="context_compression")

        # 1. Redundancy Filter
        initial_count = len(docs)
        docs = self.remove_redundancy(docs)

        # 2. Format for Lingua
        # Initial State
        raw_context = self.format_context(docs)
        tokens_before = self.count_tokens(raw_context)

        # 3. LLMLingua Compression
        compressed_context = self.compress_with_lingua(question, raw_context)
        # Final State
        tokens_after = self.count_tokens(compressed_context)
        compression_ratio = (1 - (tokens_after / tokens_before)) * 100

        print(f"\n✂️  CONTEXT COMPRESSION REPORT")
        print(f"Tokens Before: {tokens_before}")
        print(f"Tokens After:  {tokens_after}")
        print(f"Reduction:     {compression_ratio:.1f}%")

        comp_span.update(output={
            "original_doc_count": initial_count,
            "final_doc_count": len(docs),
            "tokens_before": tokens_before,
            "tokens_after": tokens_after,
            "reduction_percentage": compression_ratio,
            "compressed_context": compressed_context
        })
        comp_span.end()

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

        # response = chain.invoke({
        #     "context": compressed_context,
        #     "question": question
        # })

        full_response = ""
        print(f"\n🤖 LLM Response: ", end="", flush=True)

        # Start streaming
        for chunk in chain.stream({
            "context": compressed_context,
            "question": question
        }):
            # 1. Capture the chunk
            full_response += chunk
            # 2. Print to console immediately for better UX
            print(chunk, end="", flush=True)

            # Optional: You can yield chunk here if you want to use this in a web UI
            # yield chunk

        print("\n") # New line after stream ends

        span.update(output={"response": full_response})
        span.end()

        # -------------------------
        # Step 6: Evaluation (NEW 🔥)
        # -------------------------
        eval_span = trace.span(name="evaluation")

        claims = extract_claims(full_response)
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
            output={"final_answer": full_response},
            metadata={
                "top_k": top_k,
                "model": self.llm.model
            }
        )
        clean_response = re.sub(r'<think>.*?</think>', '', full_response, flags=re.DOTALL).strip()

        return clean_response