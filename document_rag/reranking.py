"""Lazy local cross-encoder scoring for hybrid retrieval candidates."""

from functools import lru_cache
import os


@lru_cache(maxsize=2)
def get_cross_encoder(model_name: str):
    # Keep a failed first download from holding an Ask request indefinitely.
    os.environ.setdefault("HF_HUB_ETAG_TIMEOUT", "5")
    os.environ.setdefault("HF_HUB_DOWNLOAD_TIMEOUT", "10")
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
    from sentence_transformers import CrossEncoder

    return CrossEncoder(model_name)


def rerank_scores(model_name: str, question: str, passages: list[str]) -> list[float]:
    if not passages:
        return []
    scores = get_cross_encoder(model_name).predict(
        [(question, passage) for passage in passages],
        show_progress_bar=False,
    )
    return [float(score) for score in scores]
