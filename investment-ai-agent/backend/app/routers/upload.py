#
# upload.py: Defines the API endpoint for file ingestion and processing.
# - Receives a file upload (PDF, DOCX, XLSX).
# - Uses the parsing service to extract text.
# - Chunks the extracted text.
# - Calls the RAG service to generate embeddings and store them in ChromaDB.
#
import io
import uuid
import mimetypes
from fastapi import APIRouter, UploadFile, File, HTTPException
from typing import Dict, List, Any

# Import the necessary services and schemas
from app.services import parsing, rag, llm
from app.models.schemas import UploadResponse

router = APIRouter()

# In-memory registry of uploaded documents. For a hackathon, this is a simple
# way to track what has been uploaded without a persistent database.
# The 'analyze' endpoint will check this registry to validate a document_id.
DOC_REGISTRY: Dict[str, Dict[str, Any]] = {}

@router.post("/upload", response_model=UploadResponse)
async def upload_document(file: UploadFile = File(...)):
    """
    Handles the upload of a document. This is the first step in the workflow.
    
    1.  Generates a unique ID for the document.
    2.  Reads the file content and determines its type.
    3.  Calls the appropriate parsing function to extract text.
    4.  Splits the text into smaller, manageable chunks.
    5.  Creates a unique ChromaDB collection for the document.
    6.  Generates vector embeddings for each chunk and stores them in the collection.
    7.  Returns the unique document_id to the client for use in subsequent API calls.
    """
    document_id = str(uuid.uuid4())
    filename = file.filename
    content = await file.read()
    
    # 1. Determine file type and select the correct parser
    mime_type, _ = mimetypes.guess_type(filename)
    
    pages_data: List[Dict] = []
    try:
        if mime_type == 'application/pdf':
            pages_data = parsing.extract_text_from_pdf(io.BytesIO(content))
        elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
            pages_data = parsing.extract_text_from_docx(io.BytesIO(content))
        elif mime_type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
            pages_data = parsing.extract_text_from_xlsx(io.BytesIO(content))
        else:
            # If the file type is not supported, return a clear error.
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {mime_type or 'unknown'}")
    except Exception as e:
        # Catch any errors during the parsing process.
        raise HTTPException(status_code=500, detail=f"Failed to parse document: {str(e)}")

    all_chunks, all_metadatas, all_ids = [], [], []

    # 2. Process each page and chunk its text
    for page_info in pages_data:
        page_num = page_info['page']
        page_text = page_info['text']
        
        # Skip empty pages to avoid processing blank content
        if not page_text or page_text.isspace():
            continue
        
        chunks = parsing.chunk_text(page_text)
        for i, chunk in enumerate(chunks):
            chunk_id = f"{document_id}_p{page_num}_c{i}"
            all_chunks.append(chunk)
            # Metadata is crucial for citations and context
            all_metadatas.append({"document_id": document_id, "page": page_num})
            all_ids.append(chunk_id)
    
    if not all_chunks:
        raise HTTPException(status_code=400, detail="Could not extract any text from the document.")

    # 3. Get the embedding function (handles DEMO_MODE automatically)
    embed_func = llm.get_embedder()
    
    # 4. Create a new collection and store the chunks and their embeddings
    rag.upsert_chunks(
        collection_name=document_id,  # Use the unique doc ID as the collection name
        chunks=all_chunks,
        metadatas=all_metadatas,
        ids=all_ids,
        embed_func=embed_func
    )

    # 5. Register the document as successfully processed
    DOC_REGISTRY[document_id] = {"filename": filename, "pages": len(pages_data), "chunks": len(all_chunks)}
    
    return UploadResponse(
        document_id=document_id,
        filename=filename,
        pages=len(pages_data),
        chunks=len(all_chunks),
    )