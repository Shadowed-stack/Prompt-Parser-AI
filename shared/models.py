from pydantic import BaseModel
from typing import List, Optional

class ParsedPrompt(BaseModel):
    crop: str
    location: str
    temperature: Optional[float] = None
    stress: List[str] = []

class Trait(BaseModel):
    trait_name: str
    value: float
    similarity: float

class Research(BaseModel):
    title: str
    finding: str

class Spec(BaseModel):
    crop: str
    location: str
    temperature: float
    stress: List[str]
    traits: List[str]
    scientific_basis: List[str]
    confidence: float
