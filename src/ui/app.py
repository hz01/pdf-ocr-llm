"""
Gradio Web UI for PDF OCR with LLMs.

This module provides a user-friendly web interface for processing PDFs and images
using Qwen 2.5 VL models.
"""

import gradio as gr
import tempfile
import os
from pathlib import Path
import logging

from src.pipeline.ocr_pipeline import OCRPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize OCR pipeline
pipeline = OCRPipeline("config.yaml")

# Get available models
available_models = [model['name'] for model in pipeline.list_available_models()]


def process_pdf(pdf_file, model_name, custom_prompt, dpi):
    """
    Process a PDF file and extract text.
    
    Args:
        pdf_file: Uploaded PDF file path
        model_name: Selected model name
        custom_prompt: Custom prompt (optional)
        dpi: DPI for conversion
    
    Returns:
        Extracted markdown text
    """
    if pdf_file is None:
        return "Please upload a PDF file."
    
    if not model_name:
        return "Please select a model."
    
    try:
        # Create temp output file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode='w', encoding='utf-8') as temp_output:
            temp_output_path = temp_output.name
        
        # Process PDF
        logger.info(f"Processing PDF: {pdf_file} with model: {model_name}")
        prompt = custom_prompt if custom_prompt.strip() else None
        
        result = pipeline.process_pdf(
            pdf_path=pdf_file,
            model_name=model_name,
            output_path=temp_output_path,
            prompt=prompt
        )
        
        # Read the output
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        
        # Cleanup
        os.unlink(temp_output_path)
        
        # Prepare status message
        status = f"Successfully processed {result.get('num_pages', 0)} pages in {result.get('processing_time', 0):.2f} seconds"
        
        return markdown_text, status
        
    except Exception as e:
        logger.error(f"Error processing PDF: {e}")
        return f"Error: {str(e)}", f"Processing failed: {str(e)}"


def process_image(image_file, model_name, custom_prompt):
    """
    Process an image file and extract text.
    
    Args:
        image_file: Uploaded image file path
        model_name: Selected model name
        custom_prompt: Custom prompt (optional)
    
    Returns:
        Extracted markdown text
    """
    if image_file is None:
        return "Please upload an image file."
    
    if not model_name:
        return "Please select a model."
    
    try:
        # Create temp output file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode='w', encoding='utf-8') as temp_output:
            temp_output_path = temp_output.name
        
        # Process image
        logger.info(f"Processing image: {image_file} with model: {model_name}")
        prompt = custom_prompt if custom_prompt.strip() else None
        
        result = pipeline.process_image(
            image_path=image_file,
            model_name=model_name,
            output_path=temp_output_path,
            prompt=prompt
        )
        
        # Read the output
        with open(temp_output_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()
        
        # Cleanup
        os.unlink(temp_output_path)
        
        # Prepare status message
        status = f"Successfully processed image ({len(markdown_text)} characters extracted)"
        
        return markdown_text, status
        
    except Exception as e:
        logger.error(f"Error processing image: {e}")
        return f"Error: {str(e)}", f"Processing failed: {str(e)}"


def download_markdown(markdown_text):
    """Create a downloadable markdown file."""
    if not markdown_text or markdown_text.startswith("Error") or markdown_text.startswith("Please"):
        return None
    
    with tempfile.NamedTemporaryFile(delete=False, suffix=".md", mode='w', encoding='utf-8') as f:
        f.write(markdown_text)
        return f.name


# Create Gradio interface with custom theme
custom_theme = gr.themes.Soft(
    font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
    font_mono=[gr.themes.GoogleFont("IBM Plex Mono"), "ui-monospace", "Consolas", "monospace"],
)

with gr.Blocks(title="PDF OCR with Vision Language Models", theme=custom_theme) as demo:
    gr.Markdown(
        """
        # PDF OCR with Vision Language Models
        
        Extract text from PDFs and images using state-of-the-art vision-language models.
        """
    )
    
    with gr.Tabs():
        # PDF Processing Tab
        with gr.Tab("PDF Processing"):
            with gr.Row():
                with gr.Column():
                    pdf_input = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"],
                        type="filepath"
                    )
                    pdf_model = gr.Dropdown(
                        choices=available_models,
                        label="Select Model",
                        value=available_models[0] if available_models else None,
                        info="Choose the vision-language model to use"
                    )
                    pdf_prompt = gr.Textbox(
                        label="Custom Prompt (Optional)",
                        placeholder="Leave empty for default OCR prompt...",
                        lines=3
                    )
                    pdf_button = gr.Button("Process PDF", variant="primary")
                
                with gr.Column():
                    pdf_status = gr.Textbox(label="Status", interactive=False)
                    pdf_output = gr.Textbox(
                        label="Extracted Markdown",
                        lines=20,
                        show_copy_button=True
                    )
                    pdf_download = gr.File(label="Download Result")
            
            pdf_button.click(
                fn=lambda file, model, prompt: process_pdf(file, model, prompt, 300),
                inputs=[pdf_input, pdf_model, pdf_prompt],
                outputs=[pdf_output, pdf_status]
            )
            
            pdf_output.change(
                fn=download_markdown,
                inputs=[pdf_output],
                outputs=[pdf_download]
            )
        
        # Image Processing Tab
        with gr.Tab("Image Processing"):
            with gr.Row():
                with gr.Column():
                    image_input = gr.Image(
                        label="Upload Image",
                        type="filepath"
                    )
                    image_model = gr.Dropdown(
                        choices=available_models,
                        label="Select Model",
                        value=available_models[0] if available_models else None,
                        info="Choose the vision-language model to use"
                    )
                    image_prompt = gr.Textbox(
                        label="Custom Prompt (Optional)",
                        placeholder="Leave empty for default OCR prompt...",
                        lines=3
                    )
                    image_button = gr.Button("Process Image", variant="primary")
                
                with gr.Column():
                    image_status = gr.Textbox(label="Status", interactive=False)
                    image_output = gr.Textbox(
                        label="Extracted Markdown",
                        lines=20,
                        show_copy_button=True
                    )
                    image_download = gr.File(label="Download Result")
            
            image_button.click(
                fn=process_image,
                inputs=[image_input, image_model, image_prompt],
                outputs=[image_output, image_status]
            )
            
            image_output.change(
                fn=download_markdown,
                inputs=[image_output],
                outputs=[image_download]
            )
        
        # Information Tab
        with gr.Tab("Information"):
            gr.Markdown(
                f"""
                ## Available Models
                
                {chr(10).join([f"- **{model}**" for model in available_models])}
                
                ## Features
                
                - **PDF Processing**: Convert PDF pages to images and extract text
                - **Image Processing**: Extract text from images directly
                - **Custom Prompts**: Provide specific instructions for extraction
                - **Markdown Output**: Results in clean, formatted markdown
                - **Download**: Save results as `.md` files
                
                ## Tips
                
                - Larger models (32B, 72B) provide better accuracy but require more memory
                - Smaller models (3B, 7B) are faster and use less memory
                
                ## Custom Prompts
                
                You can provide custom prompts for specific extraction tasks:
                
                - "Extract all text preserving tables and formatting"
                - "Extract only the invoice number, date, and total amount"
                - "Extract the document in JSON format with sections"
                
                ## System Information
                
                - **Pipeline Status**: Ready
                - **Available Models**: {len(available_models)}
                - **Config File**: config.yaml
                """
            )
    
    gr.Markdown(
        """
        ---
        ### Technical Details
        
        This application uses vision-language models for OCR tasks.
        Models are loaded on-demand and cached for faster subsequent processing.
        """
    )


if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )

