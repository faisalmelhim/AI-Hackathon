from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .routers import upload
from .services.rag import ensure_chroma_dir

VERSION = "0.1.0"

ensure_chroma_dir()

app = FastAPI(
    title="investment-ai-agent",
    version=VERSION,
    description="Backend service for document upload and indexing (MVP).",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
    allow_credentials=True,
)

app.include_router(upload.router)

@app.get("/health")
def health():
    return {"status": "ok", "service": "investment-ai-agent", "version": VERSION}
