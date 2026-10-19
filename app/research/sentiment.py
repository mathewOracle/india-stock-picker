"""Headline sentiment. Two classifiers, never mixed within one result (the model name is stored with it):

* FinBERT: a neural CLASSIFIER (positive/negative/neutral only: it cannot invent text). Needs the weights
  locally (`FINBERT_PATH` or the Hugging Face cache); we never try the network, so a blocked network fails fast.
* Loughran-McDonald: the standard published finance dictionary (built from SEC filings because general
  lexicons misread finance words like "liability" or "tax"). Deterministic, offline, ~28k headlines/sec.
"""

import math
import os
import re
import time
from dataclasses import dataclass
from typing import Protocol

from app.research.news import Headline

HALF_LIFE_DAYS = 3.0
MIN_HEADLINES = 5          # fewer than this -> "insufficient", never a made-up number
MIN_SOURCES = 3            # one outlet repeating itself isn't a consensus
POSITIVE_AT, NEGATIVE_AT = 0.15, -0.15
STRONG_NEGATIVE_AT = -0.30  # the level the verdict layer treats as a real warning
FINBERT = "ProsusAI/finbert"


class Classifier(Protocol):
    name: str

    def classify(self, titles: list[str]) -> list[tuple[float, float, float]]:
        """Per title: (p_positive, p_negative, p_neutral)."""


class FinBert:
    name = "finbert"

    def __init__(self) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self._torch = torch
        source = os.environ.get("FINBERT_PATH", FINBERT)
        self._tokenizer = AutoTokenizer.from_pretrained(source, local_files_only=True)
        self._model = AutoModelForSequenceClassification.from_pretrained(source, local_files_only=True).eval()
        labels = {v.lower(): k for k, v in self._model.config.id2label.items()}
        if set(labels) != {"positive", "negative", "neutral"}:
            raise RuntimeError(f"Unexpected FinBERT labels: {labels}")  # fail loudly, don't mis-map
        self._order = (labels["positive"], labels["negative"], labels["neutral"])

    def classify(self, titles: list[str]) -> list[tuple[float, float, float]]:
        out: list[tuple[float, float, float]] = []
        with self._torch.no_grad():
            for start in range(0, len(titles), 32):
                batch = self._tokenizer(titles[start:start + 32], padding=True, truncation=True,
                                        max_length=64, return_tensors="pt")
                probs = self._torch.softmax(self._model(**batch).logits, dim=-1).tolist()
                out += [tuple(p[i] for i in self._order) for p in probs]
        return out


NEGATION = re.compile(r"\b(no|not|without|never|neither|nor|fails? to|despite|avoids?|averts?)\s+\w+", re.I)


class LoughranMcDonald:
    """score = (P - N) / (P + N + 1): one sentiment word gives +/-0.5, and weak evidence stays near 0."""

    name = "loughran-mcdonald"

    def __init__(self) -> None:
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")   # pysentiment2 has an invalid-escape SyntaxWarning in its own source
            import pysentiment2
        self._lm = pysentiment2.LM()

    def classify(self, titles: list[str]) -> list[tuple[float, float, float]]:
        out = []
        for title in titles:
            # a negated word ("no loss", "fails to grow") is dropped rather than counted with the wrong sign
            counts = self._lm.get_score(self._lm.tokenize(NEGATION.sub(" ", title)))
            pos, neg = counts["Positive"], counts["Negative"]
            denom = pos + neg + 1
            out.append((pos / denom, neg / denom, 1 / denom))
        return out


def load_classifier() -> Classifier:
    """FinBERT when its weights are available locally, else Loughran-McDonald (the caller records which)."""
    try:
        return FinBert()
    except Exception:  # package missing, weights not on disk, corrupt cache ...
        return LoughranMcDonald()


@dataclass(frozen=True)
class Scored:
    title: str
    source: str
    published: float
    score: float          # p_pos - p_neg, in [-1, 1]


def score_headlines(headlines: list[Headline], classifier: Classifier,
                    cache: dict[str, float] | None = None) -> list[Scored]:
    """Score each headline once; `cache` (normalised title -> score, same model only) skips repeats."""
    cache = cache or {}
    todo = [h for h in headlines if h.key not in cache]
    fresh = {h.key: p - n for h, (p, n, _) in zip(todo, classifier.classify([h.title for h in todo]))} if todo else {}
    return [Scored(h.title, h.source, h.published, float(fresh.get(h.key, cache.get(h.key, 0.0)))) for h in headlines]


def aggregate(scored: list[Scored], model: str, now: float | None = None) -> dict:
    """Recency-weighted mean score + an explicit status. Weak evidence is labelled, not rounded up."""
    now = time.time() if now is None else now
    sources = {s.source for s in scored}
    base = {"model": model, "n": len(scored), "sources": len(sources), "score": None, "label": "No data"}
    if not scored:
        return base | {"status": "insufficient", "note": "No relevant headlines found in the last 14 days."}
    weights = [0.5 ** (max(0.0, now - s.published) / 86400 / HALF_LIFE_DAYS) for s in scored]
    mean = sum(w * s.score for w, s in zip(weights, scored)) / sum(weights)
    spread = math.sqrt(sum(w * (s.score - mean) ** 2 for w, s in zip(weights, scored)) / sum(weights))
    label = "Positive" if mean >= POSITIVE_AT else "Negative" if mean <= NEGATIVE_AT else "Neutral / mixed"
    enough = len(scored) >= MIN_HEADLINES and len(sources) >= MIN_SOURCES
    note = ("" if enough else f"Only {len(scored)} headlines from {len(sources)} sources: too thin to rely on.")
    return base | {"score": round(mean, 3), "spread": round(spread, 3), "label": label,
                   "status": "ok" if enough else "insufficient", "note": note,
                   "negative_share": round(sum(s.score <= -0.3 for s in scored) / len(scored), 3),
                   "headlines": [{"title": s.title, "source": s.source, "published": s.published,
                                  "score": round(s.score, 3)} for s in sorted(scored, key=lambda x: -x.published)[:12]]}
