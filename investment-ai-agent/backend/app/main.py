#
# main.py: The main entrypoint for the FastAPI backend application.
# - Initializes the FastAPI app.
# - Configures CORS middleware to allow frontend access.
# - Includes all the API endpoint routers.
#
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

# Import the router objects from their respective files
from app.routers import upload, analyze, memo, modeling

# --- App Initialization ---
app = FastAPI(
    title="Investment AI Agent",
    version="0.1.0",
    description="A backend service for an AI-powered investment analysis tool.",
)

# --- CORS Middleware Configuration ---
# This allows the frontend (running on localhost:5173) to make requests to the backend.
# This is crucial for a decoupled frontend/backend architecture.
origins = [
    "http://localhost:5173",  # Default Vite/React dev server
    "http://localhost:3000",  # Common alternative for Create React App
    # In production, you would add your deployed frontend's URL here.
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all HTTP methods (GET, POST, etc.)
    allow_headers=["*"],  # Allows all headers
)

# --- API Routers ---
# Include the endpoints defined in the /routers directory.
# Each router is given a prefix and tags for organization in the OpenAPI docs.
app.include_router(upload.router, prefix="/api", tags=["1. Document Upload"])
app.include_router(analyze.router, prefix="/api", tags=["2. Analysis"])
app.include_router(memo.router, prefix="/api", tags=["3. Memo Generation"])
app.include_router(modeling.router, prefix="/api", tags=["4. Financial Modeling"])

# --- Health Check Endpoint ---
@app.get("/health", tags=["Health Check"])
def health_check():
    """
    A simple endpoint to confirm that the API server is running and responsive.
    """
    return {"status": "ok", "version": "0.1.0"}