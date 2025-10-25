#
# llm.py: Interacts with Large Language Models for embedding and analysis.
#
import os
import json
import hashlib
import numpy as np
from openai import OpenAI, OpenAIError
from fastapi import HTTPException # <--- FIX: Added this import
from typing import List, Callable, Dict, Any

# --- Environment Configuration ---
DEMO_MODE = os.getenv("DEMO_MODE", "false").lower() == "true"
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# --- OpenAI Client Initialization ---
client = None
if not DEMO_MODE:
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY environment variable must be set when DEMO_MODE is false.")
    client = OpenAI(api_key=OPENAI_API_KEY)

# --- Canned Responses for Demo Mode ---
CANNED_ANALYSIS_RESPONSE = {
    "company_name": "Innovate Inc. (DEMO)",
    "sector": "Enterprise SaaS",
    "financial_metrics": {"revenue": {"value": 50, "unit": "USD millions", "year": 2024, "page": 12}},
    "key_metrics": [{"metric": "Net Revenue Retention", "value": "120%", "importance": "High", "page": 15}],
    "red_flags": [{"flag": "High dependency on a single large customer.", "severity": "Medium", "category": "Market", "page": 20}],
    "business_overview": "Innovate Inc. is a leading provider of AI-powered workflow automation platforms.",
    "competitive_position": "The company holds a strong position against its main competitor.",
    "citations": [{"page": 12, "quote": "Revenue for Fiscal Year 2024 reached $50 million..."}]
}
CANNED_MEMO_EN = "# EXECUTIVE SUMMARY (DEMO)\nInnovate Inc. shows strong potential..."
CANNED_MEMO_AR = "(DEMO) Arabic translation unavailable offline.\n\n# EXECUTIVE SUMMARY (DEMO)\nInnovate Inc. shows strong potential..."

# --- Embedding Service ---
def _hash_embedder(texts: List[str], dim: int = 768) -> List[List[float]]:
    embeddings = []
    for text in texts:
        hasher = hashlib.sha256(text.encode('utf-8'))
        seed = int.from_bytes(hasher.digest()[:4], 'big')
        rng = np.random.RandomState(seed)
        vec = rng.rand(dim) - 0.5
        norm = np.linalg.norm(vec)
        embeddings.append((vec / norm).tolist() if norm > 0 else vec.tolist())
    return embeddings

def get_embedder() -> Callable[[List[str]], List[List[float]]]:
    if DEMO_MODE:
        return _hash_embedder
    else:
        def openai_embed(texts: List[str]) -> List[List[float]]:
            if not texts: return []
            texts = [text.replace("\n", " ") for text in texts]
            response = client.embeddings.create(model="text-embedding-3-small", input=texts)
            return [embedding.embedding for embedding in response.data]
        return openai_embed

# --- LLM Analysis and Generation Services ---
def strict_json_analyze(chunks: List[Dict[str, Any]], prompt: str, doc_id_for_demo: str = None) -> Dict[str, Any]:
    if DEMO_MODE:
        return CANNED_ANALYSIS_RESPONSE if doc_id_for_demo == "doc123" else {}
    context = "\n\n".join([f"--- PAGE {chunk.get('metadata', {}).get('page', 'N/A')} ---\n{chunk['document']}" for chunk in chunks])
    messages = [{"role": "system", "content": prompt}, {"role": "user", "content": f"Here is the document context:\n{context}"}]
    for attempt in range(2):
        try:
            response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.2, response_format={"type": "json_object"})
            return json.loads(response.choices[0].message.content)
        except (json.JSONDecodeError, OpenAIError) as e:
            if attempt == 0:
                messages.append({"role": "user", "content": "Your response was not valid JSON. Respond with VALID JSON only."})
            else:
                raise ValueError("Failed to get valid JSON from LLM after 2 attempts.")
    raise ValueError("Should not be reachable.")

def generate_markdown_memo(analysis_json: str, prompt: str, language: str) -> str:
    if DEMO_MODE:
        return CANNED_MEMO_AR if language == "ar" else CANNED_MEMO_EN
    final_prompt = prompt.format(analysis_json=analysis_json)
    messages = [{"role": "system", "content": "You are a professional investment analyst."}, {"role": "user", "content": final_prompt}]
    try:
        response = client.chat.completions.create(model="gpt-4o-mini", messages=messages, temperature=0.4)
        memo_content = response.choices[0].message.content
        if language == "ar":
            trans_prompt = f"Translate the following memo into professional Arabic, preserving Markdown:\n\n{memo_content}"
            trans_messages = [{"role": "system", "content": "You are a financial translator."}, {"role": "user", "content": trans_prompt}]
            trans_response = client.chat.completions.create(model="gpt-4o-mini", messages=trans_messages, temperature=0.7)
            return trans_response.choices[0].message.content
        return memo_content
    except OpenAIError as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate memo: {e}")