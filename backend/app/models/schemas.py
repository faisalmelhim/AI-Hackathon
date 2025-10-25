from datetime import datetime
from pydantic import BaseModel


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    message: str


class DocInfo(BaseModel):
    document_id: str
    filename: str
    pages: int
    chunks: int
    created_at: datetime
