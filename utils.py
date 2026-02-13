# Utility functions for the RAG application

import os
import logging
from typing import List
import PyPDF2
from config import LOG_LEVEL

# Configure logging
logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger(__name__)


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from a PDF file with robust error handling.
    
    Args:
        pdf_path: Path to the PDF file
        
    Returns:
        Extracted text from the PDF
    """
    text = ""
    try:
        with open(pdf_path, "rb") as file:
            try:
                pdf_reader = PyPDF2.PdfReader(file)
                num_pages = len(pdf_reader.pages)
                
                if num_pages == 0:
                    logger.warning(f"PDF has no pages: {pdf_path}")
                    raise ValueError("PDF file contains no pages")
                
                extracted_pages = 0
                for page_num in range(num_pages):
                    try:
                        page = pdf_reader.pages[page_num]
                        if page is None:
                            continue
                        page_text = page.extract_text()
                        if page_text and page_text.strip():
                            text += page_text + "\n"
                            extracted_pages += 1
                    except Exception as e:
                        logger.warning(f"Could not extract page {page_num} from {pdf_path}: {e}")
                        continue
                
                if extracted_pages == 0:
                    logger.error(f"No text could be extracted from PDF: {pdf_path}")
                    raise ValueError("PDF contains no readable text")
                    
                logger.info(f"Successfully extracted {len(text)} characters from {pdf_path} ({extracted_pages} pages)")
            except PyPDF2.errors.PdfReadError as e:
                logger.error(f"PDF is corrupted or unreadable: {pdf_path}")
                raise ValueError(f"Invalid PDF file: {str(e)}")
            except ValueError:
                raise
            except Exception as e:
                logger.error(f"Error reading PDF: {pdf_path} - {e}")
                raise ValueError(f"Failed to read PDF: {str(e)}")
    except FileNotFoundError:
        logger.error(f"PDF file not found: {pdf_path}")
        raise FileNotFoundError(f"File not found: {pdf_path}")
    except ValueError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error with PDF: {pdf_path} - {e}")
        raise
    return text


def extract_text_from_txt(txt_path: str) -> str:
    """
    Extract text from a TXT file.
    
    Args:
        txt_path: Path to the TXT file
        
    Returns:
        Text content from the file
    """
    try:
        with open(txt_path, "r", encoding="utf-8") as file:
            text = file.read()
        logger.info(f"Successfully read text from {txt_path}")
        return text
    except Exception as e:
        logger.error(f"Error reading text file: {e}")
        raise


def load_document(file_path: str) -> str:
    """
    Load document content based on file extension with validation.
    
    Args:
        file_path: Path to the document file
        
    Returns:
        Document content as string
        
    Raises:
        ValueError: If file type not supported or content empty
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    _, ext = os.path.splitext(file_path)
    
    try:
        if ext.lower() == ".pdf":
            content = extract_text_from_pdf(file_path)
        elif ext.lower() in [".txt", ".md"]:
            content = extract_text_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")
        
        # Validate content
        if not content or not content.strip():
            raise ValueError(f"Document is empty or has no extractable text: {file_path}")
        
        logger.info(f"Successfully loaded document: {file_path} ({len(content)} characters)")
        return content
    except (ValueError, FileNotFoundError) as e:
        logger.error(f"Validation error for {file_path}: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error loading document {file_path}: {e}")
        raise


def split_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
    """
    Split text into overlapping chunks.
    
    Args:
        text: Text to split
        chunk_size: Size of each chunk
        overlap: Overlap between chunks
        
    Returns:
        List of text chunks
    """
    chunks = []
    start = 0
    
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks


def ensure_directory_exists(directory: str):
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        directory: Path to the directory
    """
    os.makedirs(directory, exist_ok=True)
    logger.info(f"Directory ensured: {directory}")
