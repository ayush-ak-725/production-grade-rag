from sentence_transformers import CrossEncoder
import numpy as np


class Reranker:
    def __init__(self):
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

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

        if return_scores:
            return top_docs, np.array(top_scores)

        return top_docs