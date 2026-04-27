from sentence_transformers import CrossEncoder


class Reranker:
    def __init__(self):
        # Lightweight + strong model
        self.model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

    def rerank(self, query, docs, top_k=6):
        pairs = [(query, doc.page_content) for doc in docs]

        scores = self.model.predict(pairs)
        print(f"\nReranking scores: {scores}")

        ranked = sorted(
            zip(docs, scores),
            key=lambda x: x[1],
            reverse=True
        )

        return [doc for doc, _ in ranked[:top_k]]