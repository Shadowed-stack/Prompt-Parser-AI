from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3"
    chroma_collection: str = "traits"
    max_retries: int = 3

settings = Settings()
