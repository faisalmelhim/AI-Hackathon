# llm.py: Interacts with Large Language Models for embedding and analysis.
# - Provides an embedding function that works offline (demo) or with OpenAI.
# - Provides a function to call an LLM for analysis, ensuring a JSON response.

import os
import json
import hashlib
import numpy as np
from typing import List, Dict, Callable, Any
from openai import OpenAI

# --- Environment Configuration ---
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- OpenAI Client Initialization ---
client = None
if not DEMO_MODE:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")
    client = OpenAI(api_key=OPENAI_API_KEY)

# --- Canned Response for Demo Mode ---
CANNED_ANALYSIS_RESPONSE: Dict[str, Any] = {
    "company_name": "Innovate Inc.",
    "sector": "Enterprise SaaS",
    "financial_metrics": {
        "revenue": {"value": 50, "unit": "USD millions", "year": 2024, "page": 12},
        "revenue_growth": {"value": 35, "unit": "%", "page": 12},
        "gross_margin": {"value": 85, "unit": "%", "page": 13},
        "operating_margin": {"value": 15, "unit": "%", "page": 13},
        "net_margin": {"value": 10, "unit": "%", "page": 13},
        "arr": {"value": 48, "unit": "USD millions", "page": 14},
        "customer_count": {"value": 500, "page": 15}
    },
    "key_metrics": [
        {"metric": "Net Revenue Retention", "value": "120%", "importance": "High", "page": 15},
        {"metric": "Customer Acquisition Cost", "value": "12 months payback", "importance": "Medium", "page": 16}
    ],
    "red_flags": [
        {"flag": "High dependency on a single large customer.", "severity": "Medium", "category": "Market", "page": 20}
    ],
    "business_overview": "Innovate Inc. provides a leading AI-powered workflow automation platform for large enterprises. Its main product optimizes supply chain logistics, reducing operational costs for clients.",
    "competitive_position": "The company is a strong contender in the automation space, competing with larger players but differentiating through its vertical-specific AI models and strong customer support.",
    "citations": [
        {"page": 12, "quote": "Annual revenue reached $50 million in FY2024, a 35% increase year-over-year."},
        {"page": 20, "quote": "Approximately 40% of our revenue is derived from our largest client, Acme Corp."}
    ]
}

# --- Embedding Service ---

def _hash_embed(text: str, dim: int = 384) -> List[float]:
    """Generates a deterministic, fixed-size vector from text using hashing."""
    hasher = hashlib.sha256(text.encode("utf-8"))
    seed = int.from_bytes(hasher.digest()[:4], "big")
    rng = np.random.RandomState(seed)
    vec = rng.rand(dim) - 0.5
    norm = np.linalg.norm(vec)
    return (vec / norm).tolist() if norm > 0 else vec.tolist()

def get_embedder() -> Callable[[List[str]], List[List[float]]]:
    """
    Returns the appropriate embedding function based on the environment.
    - In DEMO_MODE, returns a deterministic hashing-based embedder.
    - Otherwise, returns a function that calls the OpenAI embeddings API.
    """
    if DEMO_MODE:
        return lambda texts: [_hash_embed(text) for text in texts]
    else:
        def openai_embed(texts: List[str]) -> List[List[float]]:
            if not texts:
                return []
            response = client.embeddings.create(
                model="text-embedding-3-small",
                input=[text.replace("\n", " ") for text in texts],
            )
            return [item.embedding for item in response.data]
        return openai_embed

# --- Analysis Service ---

def strict_json_analyze(chunks: List[Dict[str, Any]], prompt: str, doc_id_for_demo: str | None = None) -> Dict[str, Any]:
    """
    Analyzes text chunks using an LLM and ensures the output is valid JSON.
    Includes a single retry attempt if JSON parsing fails.
    Args:
        chunks: A list of dicts, each with 'text' and 'page_number'.
        prompt: The system prompt to guide the LLM's analysis.
        doc_id_for_demo: Used to select a canned response in demo mode.
    Returns:
        A dictionary containing the structured analysis.
    """
    if DEMO_MODE:
        if doc_id_for_demo == "doc123":
            return CANNED_ANALYSIS_RESPONSE
        generic = dict(CANNED_ANALYSIS_RESPONSE)  # shallow copy
        generic["company_name"] = "Generic Demo Corp"
        return generic

    # Build context safely
    pages = [
        f"--- START OF PAGE {chunk.get('page_number', 'N/A')} ---\n{chunk.get('text', '')}"
        for chunk in chunks
    ]
    context = "\n\n".join(pages)
    MAX_CONTEXT_CHARS = 120_000
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]

    messages = [
        {"role": "system", "content": prompt},
        {"role": "user", "content": f"Here is the document context:\n{context}"},
    ]

    last_raw = None
    for attempt in range(2):
        try:
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=messages,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            raw = response.choices[0].message.content
            last_raw = raw
            return json.loads(raw)
        except json.JSONDecodeError as e:
            if attempt == 0:
                messages.append({"role": "assistant", "content": last_raw or "Invalid format."})
                messages.append({"role": "user", "content": "Respond with VALID JSON only. No extra text."})
                continue
            raise ValueError("Failed to get valid JSON from LLM after 2 attempts.") from e
        except Exception as e:
            if attempt == 0:
                messages.append({"role": "user", "content": "Respond with VALID JSON only. No extra text."})
                continue
            raise RuntimeError(f"LLM call failed after retry: {e}") from e
