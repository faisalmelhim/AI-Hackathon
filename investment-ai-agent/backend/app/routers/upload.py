from fastapi import APIRouter

router = APIRouter(prefix="/api/upload", tags=["upload"])

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from ..models.schemas import UploadResponse, DocInfo
# ...other imports...

router = APIRouter(prefix="/api/upload", tags=["upload"])

DOC_REGISTRY = {}

@router.post("", response_model=UploadResponse)
async def upload_file(file: UploadFile = File(...)):
    # your upload logic here
    return {"message": "ok"}
