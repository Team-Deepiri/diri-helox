from pydantic import BaseSettings


class Settings(BaseSettings):
    OPENAI_API_KEY: str | None = None
    OPENAI_MODEL: str = "gpt-4o-mini"
    CORS_ORIGIN: str = "http://localhost:5173"
    PYAGENT_API_KEY: str | None = None

    class Config:
        env_file = ".env"


settings = Settings()


