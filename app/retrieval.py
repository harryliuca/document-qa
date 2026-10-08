"""Per-request vector index with BM25/semantic reciprocal-rank fusion."""

import re

import numpy as np
from rank_bm25 import BM25Okapi

from app.models import Chunk


def tokenize(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower()) or [""]


class HybridIndex:
    def __init__(self, chunks: list[Chunk], vectors: list[list[float]]):
        self.chunks = chunks
        matrix = np.asarray(vectors, dtype=np.float32)
        self.matrix = matrix / np.maximum(np.linalg.norm(matrix, axis=1, keepdims=True), 1e-9)
        self.lexical = BM25Okapi([tokenize(c.text) for c in chunks])

    def search(self, question: str, vector: list[float], top_k: int) -> list[Chunk]:
        query = np.asarray(vector, dtype=np.float32)
        semantic = self.matrix @ (query / max(np.linalg.norm(query), 1e-9))
        lexical = self.lexical.get_scores(tokenize(question))
        scores = np.zeros(len(self.chunks))
        # Both rank lists contribute; do not reward lexical zero-score matches.
        for values, positive_only in [(semantic, False), (lexical, True)]:
            for rank, index in enumerate(np.argsort(-values, kind="stable")):
                if positive_only and values[index] <= 0:
                    continue
                scores[index] += 1 / (60 + rank + 1)
        selected = np.argsort(-scores, kind="stable")[:top_k]
        return [self.chunks[int(i)] for i in selected]
