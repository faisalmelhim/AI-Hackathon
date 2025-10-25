#
# memo.py: FastAPI router for generating investment memos.
# - POST endpoint to generate a memo from a cached analysis.
#
import os
import json
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from openai import OpenAI

# Depends on the analysis cache from the analyze router
from .analyze import ANALYSIS_CACHE 

# --- Router and Client Setup ---
router = APIRouter()
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"

client = None
if not DEMO_MODE:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# --- Pydantic Models ---
class MemoRequest(BaseModel):
    document_id: str = Field(..., description="The ID of the analyzed document.")
    language: str = Field("en", pattern="^(en|ar)$", description="Language of the memo ('en' or 'ar').")

# --- Prompt Definition ---
MEMO_PROMPT = """
You are writing a professional investment memo for an institutional investor.

Based on the following analysis data:
{analysis_json}

Write a comprehensive investment memo with these sections:

# EXECUTIVE SUMMARY (<=100 words)
# COMPANY OVERVIEW (≈200 words)
# FINANCIAL PERFORMANCE
# INVESTMENT THESIS (3–5 bullets)
# KEY RISKS (5–7 with mitigations)
# VALUATION SUMMARY
# RECOMMENDATION (Pass/Watch/Consider/Strong Buy with position sizing)

Style:
- Professional and concise.
- Use numbers and page citations where available.
- Balanced: pros/cons.
- Markdown only.
"""

CANNED_MEMO = """
# EXECUTIVE SUMMARY
Innovate Inc. presents a compelling investment case in the Enterprise SaaS sector, driven by strong 35% revenue growth to $50M ARR and high net revenue retention of 120%. The company's AI-powered workflow automation platform shows strong product-market fit. While promising, risks include high customer concentration (40% of revenue from one client) and emerging competition. We recommend a 'Consider' rating with a small initial position, pending further diligence on customer diversification and competitive moat.

# COMPANY OVERVIEW
Innovate Inc. provides a leading AI-powered workflow automation platform tailored for large enterprises, focusing on optimizing complex supply chain logistics. Founded in 2018, the company has grown to serve over 500 customers (page 15). Its core product leverages proprietary machine learning models to analyze and streamline logistical operations, resulting in significant cost reductions and efficiency gains for its clients. The company's go-to-market strategy targets Fortune 1000 companies, relying on a direct sales force.

# FINANCIAL PERFORMANCE
The company demonstrates robust financial health. FY2024 revenue reached $50 million, a 35% year-over-year increase (page 12), with an impressive gross margin of 85% (page 13). Operating and net margins stand at 15% and 10% respectively. Annual Recurring Revenue (ARR) is currently $48 million (page 14), indicating a strong subscription-based model. High net revenue retention of 120% (page 15) suggests strong customer satisfaction and successful upselling.

# INVESTMENT THESIS
- **Large & Growing Market:** The market for AI-driven process automation is expanding rapidly as enterprises seek efficiency gains.
- **Strong Product Differentiation:** Innovate Inc.'s vertical-specific AI models provide a competitive edge over more generic platforms.
- **High Customer Stickiness:** Demonstrated by a 120% net revenue retention rate, indicating low churn and expansion within existing accounts.
- **Impressive Financial Profile:** A combination of high growth, strong gross margins, and profitability is rare for a company at this stage.

# KEY RISKS
- **Customer Concentration:** High dependency on a single customer (40% of revenue) poses a significant risk. Mitigation: Management must prioritize diversifying the customer base.
- **Competitive Pressure:** The market includes well-funded competitors; Innovate Inc. must continue to innovate to maintain its lead. Mitigation: Continued R&D investment.
- **Sales Cycle Complexity:** Long enterprise sales cycles can lead to lumpy revenue recognition. Mitigation: Diversifying into mid-market could provide more predictable revenue streams.

# VALUATION SUMMARY
A detailed valuation was not performed. Based on comparable public SaaS companies with similar growth and margin profiles, a valuation multiple of 8-12x ARR could be appropriate, implying a valuation of $384M - $576M.

# RECOMMENDATION
**Consider**. We recommend a small initial investment in Innovate Inc. The company's strong technology and impressive financial metrics are highly attractive. However, the identified risks, particularly customer concentration, warrant a cautious approach.
"""

@router.post("/api/memo/generate", response_model=str, responses={200: {"content": {"text/markdown": {}}}})
def generate_memo(request: MemoRequest = Body(...)):
    """
    Generates an investment memo based on a cached document analysis.
    """
    # 1. Check for cached analysis; raise 400 if not available
    analysis = ANALYSIS_CACHE.get(request.document_id)
    if not analysis:
        raise HTTPException(
            status_code=400,
            detail="Analysis for this document_id not found. Please run analysis first."
        )

    # 2. Handle Demo Mode (offline)
    if DEMO_MODE:
        memo_content = CANNED_MEMO
        if request.language == "ar":
            return "(DEMO) Arabic translation unavailable offline.\n\n" + memo_content
        return memo_content

    # --- Real LLM Call ---
    prompt = MEMO_PROMPT.format(analysis_json=json.dumps(analysis, indent=2))
    try:
        completion = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior investment analyst writing in Markdown."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.4
        )
        memo_content = completion.choices[0].message.content

        # Translate to Arabic if requested
        if request.language == "ar":
            translation_prompt = f"Translate the following investment memo into professional, formal Arabic. Preserve the Markdown formatting:\n\n{memo_content}"
            translation_completion = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a professional translator specializing in financial documents."},
                    {"role": "user", "content": translation_prompt}
                ],
                temperature=0.7
            )
            memo_content = translation_completion.choices[0].message.content
        
        return memo_content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate memo: {e}")