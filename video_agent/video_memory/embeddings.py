# Modified from Qwen-MM-Plugins for VideoLoop; see README.md in this folder.
"""Retrieval index over memory nodes: BM25, plus dense vectors when an embedding backend is set.

The backend is ``video_memory.embed_backend`` or VIDEOLOOP_EMBED_BACKEND, "module:function", called as
``function(texts: list[str]) -> list[list[float]]``. With none, search is BM25 alone.
"""

from __future__ import annotations

import json
import math
import os
import re
import sys

import numpy as np

from ..config import get_config
from ..llm_api import _load_backend

NPZ_INDEX_FILENAME = "embeddings.npz"


def _embed_backend_spec() -> str:
    return os.environ.get("VIDEOLOOP_EMBED_BACKEND") or get_config().video_memory.embed_backend


def _embed_texts(texts: list[str]) -> np.ndarray:
    spec = _embed_backend_spec()
    if not spec:
        raise RuntimeError("no embedding backend is set")
    return np.asarray(_load_backend(spec)(texts), dtype=np.float32)


_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "the",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "have",
        "has",
        "had",
        "do",
        "does",
        "did",
        "will",
        "would",
        "shall",
        "should",
        "may",
        "might",
        "can",
        "could",
        "of",
        "in",
        "to",
        "for",
        "with",
        "on",
        "at",
        "from",
        "by",
        "as",
        "into",
        "about",
        "and",
        "or",
        "but",
        "not",
        "no",
        "it",
        "its",
        "this",
        "that",
        "these",
        "those",
        "he",
        "she",
        "they",
        "we",
        "i",
        "you",
        "me",
        "him",
        "her",
        "us",
        "的",
        "了",
        "在",
        "是",
        "我",
        "有",
        "和",
        "就",
        "不",
        "人",
        "都",
        "一",
        "一个",
        "上",
        "也",
        "很",
        "到",
        "说",
        "要",
        "去",
        "你",
        "会",
        "着",
        "没有",
        "看",
        "好",
        "自己",
        "这",
    }
)

_TOKEN_RE = re.compile(r"[\w一-鿿]+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase tokens, filtering stopwords."""
    return [t for t in _TOKEN_RE.findall(text.lower()) if t not in _STOPWORDS and len(t) > 1]


def embed_nodes(nodes: list[dict], batch_size: int = 64) -> np.ndarray | None:
    """Vectors in node order, or None with no embedding backend."""
    if not _embed_backend_spec():
        print("  No embedding backend: the index is BM25 only", file=sys.stderr)
        return None
    batches = []
    for start in range(0, len(nodes), batch_size):
        batches.append(_embed_texts([node["text"] for node in nodes[start : start + batch_size]]))
        print(f"  Embedded {min(start + batch_size, len(nodes))}/{len(nodes)} nodes", file=sys.stderr)
    return np.vstack(batches).astype(np.float32)


class EmbeddingIndex:
    def __init__(self):
        self.nodes: list[dict] = []
        self.embeddings: np.ndarray | None = None
        self._normed: np.ndarray | None = None
        self._inv_index: dict[str, list[tuple[int, int, float]]] = {}
        self._doc_lens: list[int] = []
        self._avg_dl: float = 0.0
        self._dense_disabled: bool = False

    def _embed_batch(self, texts: list[str]) -> np.ndarray:
        return _embed_texts(texts)

    def _set_embeddings(self, embeddings: np.ndarray | None):
        self.embeddings = embeddings
        self._normed = None

    def _build_sparse_index(self):
        """Build BM25 inverted index from node texts."""
        n = len(self.nodes)
        if n == 0:
            return
        doc_tokens: list[list[str]] = []
        df: dict[str, int] = {}
        for node in self.nodes:
            tokens = _tokenize(node.get("text", ""))
            doc_tokens.append(tokens)
            for t in set(tokens):
                df[t] = df.get(t, 0) + 1

        self._doc_lens = [len(dt) for dt in doc_tokens]
        self._avg_dl = sum(self._doc_lens) / n if n > 0 else 1.0

        self._inv_index = {}
        for i, tokens in enumerate(doc_tokens):
            tf_map: dict[str, int] = {}
            for t in tokens:
                tf_map[t] = tf_map.get(t, 0) + 1
            for t, tf in tf_map.items():
                idf = math.log((n - df[t] + 0.5) / (df[t] + 0.5) + 1.0)
                self._inv_index.setdefault(t, []).append((i, tf, idf))

    def _sparse_search(self, query: str) -> dict[int, float]:
        """BM25 scoring for query against all nodes. Returns {node_idx: score}."""
        tokens = _tokenize(query)
        if not tokens or not self._inv_index:
            return {}
        k1 = 1.2
        b = 0.75
        scores: dict[int, float] = {}
        for t in tokens:
            postings = self._inv_index.get(t)
            if not postings:
                continue
            for idx, tf, idf in postings:
                dl = self._doc_lens[idx]
                tf_norm = (tf * (k1 + 1)) / (tf + k1 * (1 - b + b * dl / self._avg_dl))
                scores[idx] = scores.get(idx, 0.0) + idf * tf_norm
        return scores

    def _normalized(self) -> np.ndarray:
        """Return L2-normalized embedding matrix (cached)."""
        if self._normed is None or self._normed.shape[0] != self.embeddings.shape[0]:
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._normed = self.embeddings / norms
        return self._normed

    def search(
        self,
        query: str,
        top_k: int = 10,
        node_types: list[str] | None = None,
    ) -> list[dict]:
        """Hybrid search: dense cosine + sparse BM25, fused with RRF. BM25 alone without vectors."""
        if len(self.nodes) == 0:
            return []

        indices = list(range(len(self.nodes)))
        if node_types:
            nt_lower = {t.lower() for t in node_types}
            indices = [i for i in indices if self.nodes[i].get("node_type", "").lower() in nt_lower]
        if not indices:
            return []

        q_emb = None
        if self.embeddings is not None and not self._dense_disabled:
            try:
                cand = self._embed_batch([query])[0].astype(np.float32)
                if cand.shape[0] != self.embeddings.shape[1]:
                    print(
                        f"[embed] dimension mismatch (stored={self.embeddings.shape[1]}, "
                        f"query={cand.shape[0]}); falling back to BM25-only search",
                        file=sys.stderr,
                    )
                    self._dense_disabled = True
                else:
                    q_emb = cand
            except Exception as e:
                print(f"[embed] dense backend failed ({e}); falling back to BM25-only search", file=sys.stderr)
                self._dense_disabled = True

        if q_emb is not None:
            normed = self._normalized()
            q_norm = q_emb / (np.linalg.norm(q_emb) or 1)
            cosine_scores = normed @ q_norm
            dense_ranked = sorted(
                [(i, float(cosine_scores[i])) for i in indices],
                key=lambda x: x[1],
                reverse=True,
            )
            dense_rank = {idx: rank for rank, (idx, _) in enumerate(dense_ranked)}
        else:
            cosine_scores = None
            dense_rank = {}

        sparse_scores = self._sparse_search(query)
        sparse_ranked = sorted(
            [(i, sparse_scores[i]) for i in indices if sparse_scores.get(i, 0.0) > 0],
            key=lambda x: x[1],
            reverse=True,
        )
        sparse_rank = {idx: rank for rank, (idx, _) in enumerate(sparse_ranked)}

        if not dense_rank and not sparse_rank:
            return []

        rrf_k = 60
        fused = []
        for i in indices:
            rrf_score = 0.0
            if i in sparse_rank:
                rrf_score += 1.0 / (rrf_k + sparse_rank[i])
            if i in dense_rank:
                rrf_score += 1.0 / (rrf_k + dense_rank[i])
            if rrf_score == 0:
                continue
            fused.append((i, rrf_score, float(cosine_scores[i]) if cosine_scores is not None else 0.0))
        fused.sort(key=lambda x: x[1], reverse=True)

        results = []
        for i, rrf_score, cosine in fused[:top_k]:
            node = dict(self.nodes[i])
            node["score"] = round(rrf_score, 6)
            node["cosine"] = round(cosine, 4)
            results.append(node)
        return results

    def save(self, path: str):
        tmp = f"{path}.tmp"
        with open(tmp, "wb") as f:
            np.savez(
                f,
                embeddings=self.embeddings if self.embeddings is not None else np.zeros((0, 0), dtype=np.float32),
                nodes=json.dumps(self.nodes, ensure_ascii=False),
            )
        os.replace(tmp, path)

    def load(self, path: str):
        with np.load(path, allow_pickle=False) as data:
            self._set_embeddings(data["embeddings"] if data["embeddings"].size else None)
            self.nodes = json.loads(str(data["nodes"]))
        self._build_sparse_index()


RetrievalIndex = EmbeddingIndex


def load_retrieval_index(path: str) -> EmbeddingIndex:
    index = EmbeddingIndex()
    index.load(path)
    return index


def save_retrieval_index(memory_dir: str, nodes: list[dict], embeddings: np.ndarray | None) -> str:
    """Save the index of one memory; embeddings is None for a BM25 index."""
    if embeddings is not None and (embeddings.ndim != 2 or embeddings.shape[0] != len(nodes)):
        raise ValueError("embeddings must be a 2-D matrix aligned with nodes")
    index = EmbeddingIndex()
    index.nodes = nodes
    index._set_embeddings(embeddings)
    path = os.path.join(memory_dir, NPZ_INDEX_FILENAME)
    index.save(path)
    return path
