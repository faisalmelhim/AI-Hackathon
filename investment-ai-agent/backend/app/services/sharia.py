# sharia.py: Provides Sharia compliance screening services.
# - Scans text for keywords related to non-compliant activities.
# - Determines a compliance status (Pass, Review, Fail).

import re
from typing import List, Dict, Any

# Use word boundaries to reduce false positives.
INTEREST_KEYWORDS = r"\b(interest|riba|conventional bank|loan interest|usury|usurious)\b"
ALCOHOL_KEYWORDS = r"\b(alcohol|beer|wine|spirits|liquor)\b"
GAMBLING_KEYWORDS = r"\b(gambling|casino|betting|wager|lottery|bookmaker)\b"
PROHIBITED_PRODUCTS_KEYWORDS = r"\b(pork|swine|tobacco)\b"

def screen_sharia(texts: List[str], analysis: Dict[str, Any]) -> Dict[str, Any]:
    """
    Performs a Sharia compliance screen on document text and analysis results.

    Args:
        texts: A list of raw text chunks from the document.
        analysis: The structured analysis dictionary from the LLM.

    Returns:
        A dictionary with "status" and a list of "reasons".
    """
    full_text = " ".join(texts).lower()
    reasons: List[str] = []

    # Rule 1: Core business in conventional lending ==> Fail
    if re.search(INTEREST_KEYWORDS, full_text):
        overview = (analysis or {}).get("business_overview", "").lower()
        if any(k in overview for k in ("bank", "lending", "loan", "interest-bearing", "conventional finance")):
            reasons.append("Company's core business appears to be in conventional lending or interest-based finance.")
            return {"status": "Fail", "reasons": reasons}
        else:
            reasons.append("Detected keywords related to interest/riba. Further review of revenue sources is required.")

    # Rule 2: Other prohibited lines ==> Review
    if re.search(ALCOHOL_KEYWORDS, full_text):
        reasons.append("Detected keywords related to alcohol production or sale.")
    if re.search(GAMBLING_KEYWORDS, full_text):
        reasons.append("Detected keywords related to gambling or betting activities.")
    if re.search(PROHIBITED_PRODUCTS_KEYWORDS, full_text):
        reasons.append("Detected keywords related to pork or tobacco products.")

    if reasons:
        return {"status": "Review", "reasons": reasons}

    return {"status": "Pass", "reasons": ["No explicit non-compliant activities found in the text."]}
