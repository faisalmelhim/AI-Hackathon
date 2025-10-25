#
# parsing.py: Handles text extraction from different file formats and text chunking.
#
import io
import pandas as pd
import PyPDF2
import docx
from typing import List, Dict, Any # <--- FIX: Added 'Any' here

def chunk_text(text: str, chunk_size: int = 1000) -> List[str]:
    """
    Splits a large block of text into smaller chunks of a specified token size (approximated by words).
    """
    words_per_chunk = int(chunk_size * 0.75)
    words = text.split()
    return [" ".join(words[i:i + words_per_chunk]) for i in range(0, len(words), words_per_chunk)]

def extract_text_from_pdf(file_stream: io.BytesIO) -> List[Dict[str, Any]]: # Used 'Any'
    """Extracts text and page numbers from a PDF file stream."""
    reader = PyPDF2.PdfReader(file_stream)
    return [
        {"page": i + 1, "text": page.extract_text() or ""}
        for i, page in enumerate(reader.pages)
    ]

def extract_text_from_docx(file_stream: io.BytesIO) -> List[Dict[str, Any]]: # Used 'Any'
    """Extracts text from a DOCX file stream."""
    doc = docx.Document(file_stream)
    full_text = "\n".join([para.text for para in doc.paragraphs])
    return [{"page": 1, "text": full_text}]

def extract_text_from_xlsx(file_stream: io.BytesIO) -> List[Dict[str, Any]]: # Used 'Any'
    """Extracts text from all sheets of an XLSX file stream."""
    xls = pd.ExcelFile(file_stream)
    all_text = []
    for sheet_name in xls.sheet_names:
        df = pd.read_excel(xls, sheet_name=sheet_name, header=None)
        sheet_text = df.to_string(index=False, header=False) if not df.empty else ""
        all_text.append(f"--- Sheet: {sheet_name} ---\n{sheet_text}")
    return [{"page": 1, "text": "\n\n".join(all_text)}]