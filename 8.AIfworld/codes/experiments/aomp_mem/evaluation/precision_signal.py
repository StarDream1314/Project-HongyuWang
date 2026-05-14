"""Hit-count based precision signal for plan-4 v2 trajectories."""

from __future__ import annotations

import math
from typing import Sequence

from experiments.aomp_mem.core.memory import MemoryEntry
from experiments.aomp_mem.core.retrieval import RetrievalResult


def _as_entry(item: MemoryEntry | RetrievalResult) -> MemoryEntry:
    return item.entry if isinstance(item, RetrievalResult) else item


def compute_precision_signal(entry: MemoryEntry) -> float:
    """Return a bounded retrieval-usefulness proxy from entry metadata."""
    explicit = entry.metadata.get("precision_signal")
    if explicit is not None:
        return float(max(0.0, min(1.0, float(explicit))))

    hit_count = max(0, int(entry.metadata.get("hit_count", 0) or 0))
    if hit_count <= 0:
        return float(entry.quality_score)
    return float(hit_count / (hit_count + 1.0))


def record_hit(
    entries: Sequence[MemoryEntry],
    retrieved: Sequence[MemoryEntry | RetrievalResult],
) -> None:
    """Increment hit metadata for retrieved entries in-place.

    ``entries`` is accepted to make the call site explicit about the memory
    state being updated; retrieved entries are the same mutable objects returned
    from the store, so no locked memory-store API changes are required.
    """
    if not entries or not retrieved:
        return

    entry_ids = {id(entry) for entry in entries}
    seen: set[int] = set()
    touched: list[MemoryEntry] = []
    for item in retrieved:
        entry = _as_entry(item)
        identity = id(entry)
        if identity not in entry_ids or identity in seen:
            continue
        seen.add(identity)
        hit_count = max(0, int(entry.metadata.get("hit_count", 0) or 0)) + 1
        entry.metadata["hit_count"] = hit_count
        touched.append(entry)

    total_hits = sum(max(0, int(entry.metadata.get("hit_count", 0) or 0)) for entry in entries)
    if total_hits <= 0:
        return
    pool_penalty = math.sqrt(max(1, len(entries)))
    for entry in touched:
        hit_count = max(0, int(entry.metadata.get("hit_count", 0) or 0))
        entry.metadata["precision_signal"] = float((hit_count / total_hits) / pool_penalty)
