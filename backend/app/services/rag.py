def ensure_chroma_dir() -> str:
    """Create CHROMA_DIR if missing and return its path."""
    import os
    path = os.getenv("CHROMA_DIR", "./chroma")
    os.makedirs(path, exist_ok=True)
    return path
