from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


@dataclass(frozen=True)
class Settings:
    embedding_model: str
    generation_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    data_dir: Path
    storage_dir: Path

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            embedding_model=os.getenv("AEM2_EMBEDDING_MODEL", "text-embedding-3-small"),
            generation_model=os.getenv("AEM2_GENERATION_MODEL", "gpt-5.6-luna"),
            chunk_size=int(os.getenv("AEM2_CHUNK_SIZE", "120")),
            chunk_overlap=int(os.getenv("AEM2_CHUNK_OVERLAP", "24")),
            top_k=int(os.getenv("AEM2_TOP_K", "4")),
            data_dir=ROOT / "data",
            storage_dir=ROOT / "storage",
        )


def api_key() -> str:
    value = os.getenv("OPENAI_API_KEY")
    if not value:
        raise EnvironmentError("Falta OPENAI_API_KEY. Copiá .env.example a .env.")
    return value

