#
# schemas.py: Defines Pydantic models for API request and response validation.
#
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int

class AnalysisRequest(BaseModel):
    k: int = Field(12, gt=0, le=50, description="Number of chunks to retrieve for analysis.")
    language: str = Field("en", description="Language hint for analysis.")

class MemoRequest(BaseModel):
    document_id: str
    language: str = Field("en", pattern="^(en|ar)$", description="Target language for the memo.")

class DCFRequest(BaseModel):
    current_revenue: float
    growth_rates: List[float] = Field(..., min_length=5, max_length=5)
    operating_margin: float
    tax_rate: float = 0.25
    capex_percent: float = 0.05
    nwc_percent: float = 0.10
    discount_rate: float = 0.12
    terminal_growth: float = 0.03

class DCFYearlyProjection(BaseModel):
    year: int
    revenue: float
    ebit: float
    fcf: float

class DCFResponse(BaseModel):
    base: float
    bull: float
    bear: float
    yearly: List[DCFYearlyProjection]
    assumptions_used: Dict[str, Any]