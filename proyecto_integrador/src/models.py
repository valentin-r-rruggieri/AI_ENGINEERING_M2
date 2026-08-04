from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class StoredChunk(BaseModel):
    chunk_id: str
    content: str = Field(min_length=1)
    source: str
    start_word: int = Field(ge=0)
    end_word: int = Field(gt=0)
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(StoredChunk):
    score: float


class RAGAnswer(BaseModel):
    user_question: str = Field(min_length=1)
    system_answer: str = Field(min_length=1)
    chunks_related: list[RetrievedChunk] = Field(min_length=1, max_length=5)


class EvaluationResult(BaseModel):
    score: int = Field(ge=0, le=10)
    reason: str = Field(min_length=50)

