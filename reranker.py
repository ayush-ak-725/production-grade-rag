from sentence_transformers import CrossEncoder
import numpy as np
import tiktoken


class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
        self.tokenizer = tiktoken.get_encoding("cl100k_base")

    def count_tokens(self, text):
        return len(self.tokenizer.encode(text))

    def rerank(self, query, docs, top_k=6, return_scores=False):
        pairs = [(query, doc.page_content) for doc in docs]

        scores = self.model.predict(pairs)
        print(f"\nReranking scores: {scores}")

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        top_docs = [doc for doc, _ in ranked[:top_k]]
        top_scores = [score for _, score in ranked[:top_k]]
        print(f"\nTop {top_k} docs after reranking:")
        total_tokens = 0
        for i, (doc, score) in enumerate(ranked[:top_k]):
            tokens = self.count_tokens(doc.page_content)
            total_tokens += tokens

            print(
                f"{i+1}. {doc.metadata.get('source')} → {score:.4f} "
                f"| tokens: {tokens}"
            )

        print(f"\n🔥 Total tokens (context only): {total_tokens}")

        if return_scores:
            return top_docs, np.array(top_scores)

        return top_docs