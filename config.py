import os
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

load_dotenv()

os.environ.setdefault("GOOGLE_API_KEY", "")


class Settings(BaseSettings):
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    EMBEDDING_MODEL: str = "models/gemini-embedding-001"
    LLM_MODEL: str = "gemini-2.5-flash"
    VECTOR_TOP_K: int = 4

    model_config = {"env_prefix": "", "extra": "ignore"}


settings = Settings()
