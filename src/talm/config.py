"""Configuration loader and dependency factory.

Reads config.yaml, builds TALMConfig, and instantiates the correct
adapters (LLM, Embedder, VectorDB, Sandbox) based on provider settings.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import yaml

from talm.core.entities import TALMConfig
from talm.core.interfaces import IEmbedder, ILLMClient, ISandbox, IVectorDatabase

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent.parent / "config.yaml"


def load_config(path: str | Path | None = None) -> TALMConfig:
    """Load and parse config.yaml into a TALMConfig dataclass."""
    config_path = Path(path) if path else _DEFAULT_CONFIG_PATH
    if not config_path.exists():
        logger.warning("Config file not found at %s, using defaults", config_path)
        return TALMConfig()

    with open(config_path) as f:
        raw = yaml.safe_load(f)

    tree = raw.get("tree", {})
    validation = raw.get("validation", {})
    memory = raw.get("memory", {})
    llm = raw.get("llm", {})
    embedding = raw.get("embedding", {})
    vector_db = raw.get("vector_db", {})
    sandbox = raw.get("sandbox", {})

    return TALMConfig(
        max_depth=tree.get("max_depth", 3),
        initial_branching=tree.get("initial_branching", 3),
        decay_rate=tree.get("decay_rate", 1),
        max_retries=validation.get("max_retries", 3),
        similarity_threshold=memory.get("similarity_threshold", 0.75),
        top_k=memory.get("top_k", 3),
        merge_threshold=memory.get("merge_threshold", 0.95),
        llm_provider=llm.get("provider", "gemini"),
        llm_model=llm.get("model", "gemini-2.0-flash"),
        temperature=llm.get("temperature", 0.0),
        max_output_tokens=llm.get("max_output_tokens", 8192),
        embedding_provider=embedding.get("provider", "sentence_transformers"),
        embedding_model=embedding.get("model", "all-MiniLM-L6-v2"),
        vector_db_provider=vector_db.get("provider", "zvec"),
        vector_db_persist_dir=vector_db.get("persist_dir", "./talm_vector_db"),
        sandbox_timeout=sandbox.get("timeout_seconds", 30),
        sandbox_enabled=sandbox.get("enabled", True),
    )


def create_llm(config: TALMConfig) -> ILLMClient:
    """Factory: instantiate the configured LLM client."""
    if config.llm_provider == "gemini":
        from talm.infrastructure.llm_gemini import GeminiLLMClient
        return GeminiLLMClient(model_name=config.llm_model)
    raise ValueError(f"Unsupported LLM provider: {config.llm_provider}")


def create_embedder(config: TALMConfig) -> IEmbedder:
    """Factory: instantiate the configured embedding model."""
    if config.embedding_provider == "sentence_transformers":
        from talm.infrastructure.embedder_st import SentenceTransformerEmbedder
        return SentenceTransformerEmbedder(model_name=config.embedding_model)
    if config.embedding_provider == "gemini":
        from talm.infrastructure.embedder_gemini import GeminiEmbedder
        return GeminiEmbedder()
    raise ValueError(f"Unsupported embedding provider: {config.embedding_provider}")


def create_vector_db(config: TALMConfig, embedder: IEmbedder) -> IVectorDatabase:
    """Factory: instantiate the configured vector database."""
    if config.vector_db_provider == "zvec":
        from talm.infrastructure.vector_db_zvec import ZvecAdapter
        return ZvecAdapter(
            persist_dir=config.vector_db_persist_dir,
            embedding_dim=embedder.get_dimension(),
        )
    if config.vector_db_provider == "chroma":
        from talm.infrastructure.vector_db import ChromaDBAdapter
        return ChromaDBAdapter(persist_dir=config.vector_db_persist_dir)
    raise ValueError(f"Unsupported vector DB provider: {config.vector_db_provider}")


def create_sandbox(config: TALMConfig) -> ISandbox:
    """Factory: instantiate the sandbox environment."""
    from talm.infrastructure.sandbox import SubprocessSandbox
    return SubprocessSandbox(timeout=config.sandbox_timeout)
