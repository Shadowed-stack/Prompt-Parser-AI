from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "deepseek-r1:7b"

    chroma_collection: str = "crop_traits"

    max_retries: int = 3
    search_top_k: int = 10

settings = Settings()
