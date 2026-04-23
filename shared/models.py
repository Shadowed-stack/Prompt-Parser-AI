from pydantic import BaseModel, Field
from typing import List, Optional

class ParsedPrompt(BaseModel):
    crop: str
    location: str
    temperature: Optional[float]
    stress: List[str]

class Trait(BaseModel):
    trait_name: str
    value: float
    similarity_score: float

class ResearchInsight(BaseModel):
    title: str
    key_finding: str
    relevance: float

class Spec(BaseModel):
    crop: str
    location: str
    temperature: float
    stress: List[str]
    traits: List[str]
    scientific_basis: List[str]
    confidence: float = Field(..., ge=0, le=1)
