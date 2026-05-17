"""Core package for the A-OMP-Mem Evo-Memory adaptation."""

from experiments.aomp_mem.core.drift import detect_performance_drift, detect_similarity_warning
from experiments.aomp_mem.core.executor import AOMPMemExecutor
from experiments.aomp_mem.core.llm_interface import CallTracker, MockLLMClient
from experiments.aomp_mem.core.memory import FeedbackRecord, MemoryEntry, MemoryStore
from experiments.aomp_mem.core.refinement import DeepRefinementEngine, RefinementResult
from experiments.aomp_mem.core.retrieval import HashingEmbedder, RetrievalResult, retrieve_top_k

__all__ = [
    "AOMPMemExecutor",
    "CallTracker",
    "DeepRefinementEngine",
    "FeedbackRecord",
    "HashingEmbedder",
    "MemoryEntry",
    "MemoryStore",
    "MockLLMClient",
    "RefinementResult",
    "RetrievalResult",
    "detect_performance_drift",
    "detect_similarity_warning",
    "retrieve_top_k",
]
