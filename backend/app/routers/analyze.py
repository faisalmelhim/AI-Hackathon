#
# analyze.py: FastAPI router for document analysis.
# - POST endpoint to trigger a new analysis.
# - GET endpoint to retrieve a cached analysis result.
#
import os
from typing import Optional, Dict
from fastapi import APIRouter, HTTPException, Body, Path

# Assuming these modules from other developers/files exist
from app.services import llm, sharia, rag
from app.routers import upload # To access DOC_REGISTRY
from pydantic import BaseModel, Field

# --- Router Setup ---
router = APIRouter()

# --- In-Memory Cache for Analysis Results ---
ANALYSIS_CACHE: Dict[str, Dict] = {}

# --- Pydantic Models ---
class AnalysisRequest(BaseModel):
    k: int = Field(12, gt=0, le=50, description="Number of text chunks to retrieve.")
    language: str = Field("en", description="Language for analysis (currently unused in analysis).")

# --- Prompt Definition ---
ANALYSIS_PROMPT = """
You are an expert investment analyst reviewing a company document. Extract the following in valid JSON:

{
  "company_name": "string|null",
  "sector": "string|null",
  "financial_metrics": {
    "revenue": {"value": number|null, "unit": "string|null", "year": number|null, "page": number|null},
    "revenue_growth": {"value": number|null, "unit": "%", "page": number|null},
    "gross_margin": {"value": number|null, "unit": "%", "page": number|null},
    "operating_margin": {"value": number|null, "unit": "%", "page": number|null},
    "net_margin": {"value": number|null, "unit": "%", "page": number|null},
    "arr": {"value": number|null, "unit": "string|null", "page": number|null},
    "customer_count": {"value": number|null, "page": number|null}
  },
  "key_metrics": [
    {"metric": "string", "value": "string", "importance": "High|Medium|Low", "page": number|null}
  ],
  "red_flags": [
    {"flag": "string", "severity": "High|Medium|Low", "category": "Financial|Legal|Operational|Market|Sharia", "page": number|null}
  ],
  "business_overview": "2-3 sentence summary",
  "competitive_position": "1-2 sentence summary",
  "citations": [{"page": number, "quote": "string"}]
}

Rules:
- Only include metrics explicitly found; otherwise use null.
- Provide page numbers when possible; else null.
- Be conservative in red-flag severity.
- Output strict JSON only, no extra text.
"""


@router.post("/api/analyze/{document_id}", status_code=200)
def create_analysis(
    document_id: str = Path(..., description="The ID of the uploaded document."),
    request: Optional[AnalysisRequest] = Body(default=None)
):
    """
    Triggers a financial and Sharia analysis of a document. Caches the result.
    """
    # Use default request model if the body is empty or not provided
    req_body = request if request is not None else AnalysisRequest()
        
    # 1. Validate document exists in the registry from the upload router
    if document_id not in upload.DOC_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Document with ID '{document_id}' not found.")

    try:
        # 2. Retrieve top-k chunks using the RAG service
        embed_func = llm.get_embedder()
        # Per requirements, an empty query_text is acceptable for MVP
        top_chunks = rag.get_top_k(document_id, query_text="", k=req_body.k, embed_func=embed_func)
        
        if not top_chunks:
            raise HTTPException(status_code=400, detail="Could not retrieve text chunks from the document.")

        # 3. Call LLM for structured analysis
        analysis_result = llm.strict_json_analyze(top_chunks, ANALYSIS_PROMPT, doc_id_for_demo=document_id)
        
        # 4. Run Sharia screening on the raw texts and the LLM analysis
        chunk_texts = [chunk.get('text', '') for chunk in top_chunks]
        sharia_findings = sharia.screen_sharia(chunk_texts, analysis_result)
        
        # 5. Add sharia_findings object to the final result
        analysis_result["sharia_findings"] = sharia_findings
        
        # Append Sharia findings to red_flags if status is not "Pass"
        if sharia_findings["status"] != "Pass":
            severity_map = {"Fail": "High", "Review": "Medium"}
            severity = severity_map.get(sharia_findings["status"], "Low")
            for reason in sharia_findings["reasons"]:
                if "No explicit non-compliant" in reason: continue
                analysis_result["red_flags"].append({
                    "flag": reason,
                    "severity": severity,
                    "category": "Sharia",
                    "page": None # Page number is hard to determine for cross-chunk keywords
                })
        
        # 6. Cache and return the result
        ANALYSIS_CACHE[document_id] = analysis_result
        return analysis_result

    except Exception as e:
        # Catch potential errors from services (LLM, RAG)
        print(f"An error occurred during analysis for doc {document_id}: {e}")
        raise HTTPException(status_code=500, detail=f"An internal error occurred during analysis: {e}")


@router.get("/api/analyze/{document_id}")
def get_analysis(document_id: str = Path(..., description="The ID of the document to retrieve analysis for.")):
    """
    Retrieves a cached analysis result for a given document ID.
    """
    result = ANALYSIS_CACHE.get(document_id)
    if not result:
        raise HTTPException(
            status_code=404,
            detail="Analysis not found. Please run the analysis first via a POST request."
        )
    return result