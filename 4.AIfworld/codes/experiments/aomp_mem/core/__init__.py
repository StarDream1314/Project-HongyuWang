"""Core utilities for the A-OMP-Mem experiment package."""

from experiments.aomp_mem.core.drift import detect_performance_drift, detect_similarity_warning
from experiments.aomp_mem.core.embedding import EmbedderProtocol, SentenceTransformerEmbedder
from experiments.aomp_mem.core.executor import AOMPMemExecutor
from experiments.aomp_mem.core.llm_interface import (
    CallTracker,
    GeminiHTTPClient,
    MockLLMClient,
    OpenAIHTTPClient,
    RemoteLLMError,
    build_gemini_client_from_env,
    build_openai_client_from_env,
)
from experiments.aomp_mem.core.memory import FeedbackRecord, MemoryEntry, MemoryStore
from experiments.aomp_mem.core.refinement import DeepRefinementEngine, RefinementResult
from experiments.aomp_mem.core.retrieval import HashingEmbedder, RetrievalResult, retrieve_top_k

__all__ = [
    "AOMPMemExecutor",
    "CallTracker",
    "DeepRefinementEngine",
    "EmbedderProtocol",
    "FeedbackRecord",
    "GeminiHTTPClient",
    "HashingEmbedder",
    "MemoryEntry",
    "MemoryStore",
    "MockLLMClient",
    "OpenAIHTTPClient",
    "RefinementResult",
    "RemoteLLMError",
    "RetrievalResult",
    "SentenceTransformerEmbedder",
    "build_gemini_client_from_env",
    "build_openai_client_from_env",
    "detect_performance_drift",
    "detect_similarity_warning",
    "retrieve_top_k",
]
