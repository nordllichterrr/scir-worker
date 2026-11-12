import math
import re
from collections import Counter
from typing import List, Dict, Tuple


_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> List[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


class BM25Index:


    def __init__(self, docs: List[str], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b

        self._docs_tokens: List[Counter] = []
        self._doc_lengths: List[int] = []
        self._df: Counter = Counter()

        for text in docs:
            tokens = _tokenize(text)
            freqs = Counter(tokens)
            self._docs_tokens.append(freqs)
            self._doc_lengths.append(len(tokens))
            for term in freqs.keys():
                self._df[term] += 1

        self.N = len(self._docs_tokens)
        self.avgdl = (sum(self._doc_lengths) / self.N) if self.N > 0 else 0.0


        self._idf: Dict[str, float] = {}
        for term, df in self._df.items():
            self._idf[term] = math.log(1.0 + (self.N - df + 0.5) / (df + 0.5))

    def _score_single(self, query_tokens: List[str]) -> List[float]:
        scores = [0.0] * self.N
        if not query_tokens or self.N == 0 or self.avgdl == 0:
            return scores

        for term in query_tokens:
            idf = self._idf.get(term)
            if idf is None:
                continue

            for idx, freqs in enumerate(self._docs_tokens):
                f = freqs.get(term)
                if not f:
                    continue
                dl = self._doc_lengths[idx]
                denom = f + self.k1 * (1.0 - self.b + self.b * dl / self.avgdl)
                scores[idx] += idf * (f * (self.k1 + 1.0) / denom)

        return scores

    def score(self, query: str) -> List[float]:
        tokens = _tokenize(query)
        return self._score_single(tokens)

    def search(self, query: str, top_k: int = 100) -> Tuple[List[int], List[float]]:

        scores = self.score(query)
        if not scores:
            return [], []

        top_k = min(top_k, len(scores))

        idxs = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return idxs, [scores[i] for i in idxs]

