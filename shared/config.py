"""
Shared settings for PRANAG-AI.

Changes over previous version:
  • Removed ChromaDB settings (chroma_collection, chroma_persist_dir)
    — ChromaDB is no longer used, replaced by parquet-native numpy search
  • Removed Ollama settings (commented out block kept for reference)
  • Removed semantic_scholar settings — pipeline uses OpenAlex (free, no key needed)
  • Added parquet_path setting so the path is configured in one place
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):

    # ── LLM (OpenRouter) ──────────────────────────────────────────────────────
    openrouter_api_key:  Optional[str] = None
    openrouter_base_url: str           = "https://openrouter.ai/api/v1"
    openrouter_model:    str           = "openai/gpt-4o"
    llm_timeout:         int           = 30

    # ── Vector search (parquet-native numpy — no DB server needed) ────────────
    # Path to universal_index_final.parquet, relative to Prompt_Parser folder.
    # Override in .env:  PARQUET_PATH=../PINN Framework/datasrc/universal_index_final.parquet
    parquet_path: str = "../PINN Framework/datasrc/universal_index_final.parquet"

    # ── Pipeline ──────────────────────────────────────────────────────────────
    max_retries:  int = 1
    search_top_k: int = 10

    # ── Research (OpenAlex — free, no API key required) ───────────────────────
    research_results_limit: int = 5

    # ── Output ────────────────────────────────────────────────────────────────
    pipeline_version: str           = "1.0.0"
    output_dir:       str           = "./outputs"
    srikar_endpoint:  Optional[str] = None
    export_to_file:   bool          = True

    model_config = {"env_file": ".env", "extra": "ignore"}


settings = Settings()