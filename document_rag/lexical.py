"""Small in-memory BM25 index for the currently active document set."""

from collections import Counter
from dataclasses import dataclass
from math import log
import re

from document_rag.models import DocumentChunk


_TOKEN_PATTERN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN_PATTERN.findall(text.casefold())


@dataclass(frozen=True)
class LexicalIndex:
    chunks: tuple[DocumentChunk, ...]
    frequencies: tuple[Counter[str], ...]
    lengths: tuple[int, ...]
    document_frequency: Counter[str]
    average_length: float


def build_lexical_index(chunks: list[DocumentChunk]) -> LexicalIndex:
    items = tuple(chunks)
    frequencies = tuple(Counter(tokenize(chunk.embedding_text)) for chunk in items)
    document_frequency: Counter[str] = Counter()
    for counts in frequencies:
        document_frequency.update(counts.keys())
    lengths = tuple(sum(counts.values()) for counts in frequencies)
    return LexicalIndex(
        chunks=items,
        frequencies=frequencies,
        lengths=lengths,
        document_frequency=document_frequency,
        average_length=sum(lengths) / len(items) if items else 0.0,
    )


def search_lexical(index: LexicalIndex, query: str, limit: int) -> list[tuple[DocumentChunk, float]]:
    """Rank exact term matches with BM25; keep scores separate from vector distances."""
    terms = set(tokenize(query))
    if not terms or not index.chunks:
        return []
    size = len(index.chunks)
    average_length = index.average_length or 1.0
    matches: list[tuple[DocumentChunk, float]] = []
    for chunk, frequencies, length in zip(index.chunks, index.frequencies, index.lengths):
        score = 0.0
        for term in terms:
            frequency = frequencies.get(term, 0)
            if not frequency:
                continue
            document_frequency = index.document_frequency[term]
            inverse_frequency = log(1 + (size - document_frequency + 0.5) / (document_frequency + 0.5))
            score += inverse_frequency * (frequency * 2.2) / (
                frequency + 1.2 * (0.25 + 0.75 * length / average_length)
            )
        if score > 0:
            matches.append((chunk, score))
    return sorted(matches, key=lambda item: (-item[1], item[0].chunk_id))[:limit]
