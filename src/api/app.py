"""
FastAPI REST API for PDF OCR with LLMs.

This module provides a REST API for processing PDFs and images using Qwen 2.5 VL models.
"""

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from typing import Optional, List
import tempfile
import os
from pathlib import Path
import logging
import shutil

from src.pipeline.ocr_pipeline import OCRPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="PDF OCR with LLMs API",
    description="REST API for extracting text from PDFs using Qwen 2.5 VL models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize OCR pipeline
pipeline = None


@app.on_event("startup")
async def startup_event():
    """Initialize the OCR pipeline on startup."""
    global pipeline
    logger.info("Initializing OCR Pipeline...")
    pipeline = OCRPipeline("config.yaml")
    logger.info("OCR Pipeline initialized successfully")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    global pipeline
    if pipeline:
        pipeline.cleanup()
    logger.info("Application shutdown complete")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "PDF OCR with LLMs API",
        "version": "1.0.0",
        "endpoints": {
            "models": "/models",
            "process_pdf": "/process/pdf",
            "process_image": "/process/image",
            "batch_process": "/process/batch"
        }
    }


@app.get("/models")
async def list_models():
    """List all available models."""
    try:
        models = pipeline.list_available_models()
        return {"models": models}
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/process/pdf")
async def process_pdf(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    prompt: Optional[str] = Form(None)
):
    """
    Process a PDF file and extract text.
    
    Args:
        file: PDF file to process
        model_name: Name of the model to use
        prompt: Optional custom prompt
    
    Returns:
        Extracted text in markdown format
    """
    temp_pdf = None
    temp_output = None
    
    try:
        # Save uploaded file to temp location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_pdf:
            shutil.copyfileobj(file.file, temp_pdf)
            temp_pdf_path = temp_pdf.name
        
        # Create temp output file
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
        temp_output_path = temp_output.name
        temp_output.close()
        
        # Process PDF
        logger.info(f"Processing PDF with model: {model_name}")
        result = pipeline.process_pdf(
            pdf_path=temp_pdf_path,
            model_name=model_name,
            output_path=temp_output_path,
            prompt=prompt
        )
        
        # Read the output
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        
        return {
            "success": True,
            "model": model_name,
            "num_pages": result.get("num_pages", 0),
            "markdown": markdown_text,
            "processing_time": result.get("processing_time", 0)
        }
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup temp files
        if temp_pdf and os.path.exists(temp_pdf_path):
            os.unlink(temp_pdf_path)
        if temp_output and os.path.exists(temp_output_path):
            os.unlink(temp_output_path)


@app.post("/process/image")
async def process_image(
    file: UploadFile = File(...),
    model_name: str = Form(...),
    prompt: Optional[str] = Form(None)
):
    """
    Process an image file and extract text.
    
    Args:
        file: Image file to process
        model_name: Name of the model to use
        prompt: Optional custom prompt
    
    Returns:
        Extracted text in markdown format
    """
    temp_image = None
    temp_output = None
    
    try:
        # Save uploaded file to temp location
        suffix = Path(file.filename).suffix
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_image:
            shutil.copyfileobj(file.file, temp_image)
            temp_image_path = temp_image.name
        
        # Create temp output file
        temp_output = tempfile.NamedTemporaryFile(delete=False, suffix=".md")
        temp_output_path = temp_output.name
        temp_output.close()
        
        # Process image
        logger.info(f"Processing image with model: {model_name}")
        result = pipeline.process_image(
            image_path=temp_image_path,
            model_name=model_name,
            output_path=temp_output_path,
            prompt=prompt
        )
        
        # Read the output
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        
        return {
            "success": True,
            "model": model_name,
            "markdown": markdown_text,
            "text_length": len(markdown_text)
        }
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Cleanup temp files
        if temp_image and os.path.exists(temp_image_path):
            os.unlink(temp_image_path)
        if temp_output and os.path.exists(temp_output_path):
            os.unlink(temp_output_path)


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "pipeline_ready": pipeline is not None
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

