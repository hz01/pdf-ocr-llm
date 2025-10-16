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
import zipfile
from datetime import datetime
import torch
import transformers
import sys

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

# Gather system information
def get_system_info():
    """Collect system and library information."""
    info = {}
    
    # Python version
    info['python_version'] = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    
    # CUDA availability
    info['cuda_available'] = torch.cuda.is_available()
    
    # GPU information
    if info['cuda_available']:
        info['cuda_version'] = torch.version.cuda
        info['gpu_count'] = torch.cuda.device_count()
        info['gpus'] = []
        for i in range(info['gpu_count']):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)  # GB
            info['gpus'].append({
                'id': i,
                'name': gpu_name,
                'memory': f"{gpu_memory:.1f} GB"
            })
    else:
        info['cuda_version'] = "N/A"
        info['gpu_count'] = 0
        info['gpus'] = []
    
    # Library versions
    info['pytorch_version'] = torch.__version__
    info['transformers_version'] = transformers.__version__
    
    try:
        import gradio
        info['gradio_version'] = gradio.__version__
    except:
        info['gradio_version'] = "Unknown"
    
    try:
        import accelerate
        info['accelerate_version'] = accelerate.__version__
    except:
        info['accelerate_version'] = "Not installed"
    
    try:
        import PIL
        info['pillow_version'] = PIL.__version__
    except:
        info['pillow_version'] = "Unknown"
    
    return info

system_info = get_system_info()


def process_pdf(pdf_file, model_name, custom_prompt, temperature, top_p, max_tokens):
    """
    Process a PDF file and extract text.
    
    Args:
        pdf_file: Uploaded PDF file path
        model_name: Selected model name
        custom_prompt: Custom prompt (optional)
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Maximum output tokens
    
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
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=int(max_tokens)
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


def process_image(image_file, model_name, custom_prompt, temperature, top_p, max_tokens):
    """
    Process an image file and extract text.
    
    Args:
        image_file: Uploaded image file path
        model_name: Selected model name
        custom_prompt: Custom prompt (optional)
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Maximum output tokens
    
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
            prompt=prompt,
            temperature=temperature,
            top_p=top_p,
            max_new_tokens=int(max_tokens)
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


def process_batch_pdfs(pdf_files, model_name, temperature, top_p, max_tokens, progress=gr.Progress()):
    """
    Process multiple PDF files and create a zip with results.
    
    Args:
        pdf_files: List of uploaded PDF files
        model_name: Selected model name
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Maximum output tokens
        progress: Gradio progress tracker
    
    Returns:
        Tuple of (zip_file_path, status_message)
    """
    if not pdf_files or len(pdf_files) == 0:
        return None, "Please upload at least one PDF file."
    
    if not model_name:
        return None, "Please select a model."
    
    try:
        # Create temp directory for outputs
        output_dir = tempfile.mkdtemp()
        results = []
        total_files = len(pdf_files)
        
        logger.info(f"Processing {total_files} PDFs with model: {model_name}")
        
        # Process each PDF
        for idx, pdf_file in enumerate(pdf_files, 1):
            # Update progress
            progress(idx / total_files, desc=f"Processing {idx}/{total_files}: {Path(pdf_file.name).name}")
            
            try:
                # Get original filename without extension
                pdf_name = Path(pdf_file.name).stem
                output_path = os.path.join(output_dir, f"{pdf_name}.md")
                
                # Process PDF
                result = pipeline.process_pdf(
                    pdf_path=pdf_file.name,
                    model_name=model_name,
                    output_path=output_path,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=int(max_tokens)
                )
                
                results.append({
                    'file': pdf_name,
                    'pages': result.get('num_pages', 0),
                    'status': 'Success'
                })
                
            except Exception as e:
                logger.error(f"Error processing {pdf_file.name}: {e}")
                results.append({
                    'file': Path(pdf_file.name).stem,
                    'pages': 0,
                    'status': f'Failed: {str(e)}'
                })
        
        # Create zip file
        progress(0.95, desc="Creating ZIP file...")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        zip_path = os.path.join(output_dir, f"ocr_results_{timestamp}.zip")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
            for md_file in Path(output_dir).glob("*.md"):
                zipf.write(md_file, md_file.name)
        
        # Prepare status message
        progress(1.0, desc="Complete!")
        success_count = sum(1 for r in results if r['status'] == 'Success')
        total_pages = sum(r['pages'] for r in results)
        
        status = f"✓ Processed {success_count}/{total_files} PDFs successfully\n"
        status += f"✓ Total pages: {total_pages}\n\n"
        status += "Results:\n"
        for r in results:
            icon = "✓" if r['status'] == 'Success' else "✗"
            status += f"{icon} {r['file']}: {r['status']}"
            if r['pages'] > 0:
                status += f" ({r['pages']} pages)"
            status += "\n"
        
        return zip_path, status
        
    except Exception as e:
        logger.error(f"Error in batch processing: {e}")
        return None, f"✗ Batch processing failed: {str(e)}"


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
                    
                    # Generation Parameters (Collapsible)
                    with gr.Accordion("Generation Parameters", open=False):
                        pdf_temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.1,
                            step=0.1,
                            label="Temperature",
                            info="Controls randomness (0=deterministic, 1=creative)"
                        )
                        pdf_top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top P",
                            info="Nucleus sampling threshold"
                        )
                        pdf_max_tokens = gr.Slider(
                            minimum=128,
                            maximum=4096,
                            value=2048,
                            step=128,
                            label="Max Output Tokens",
                            info="Maximum length of generated text"
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
                fn=process_pdf,
                inputs=[pdf_input, pdf_model, pdf_prompt, pdf_temperature, pdf_top_p, pdf_max_tokens],
                outputs=[pdf_output, pdf_status]
            )
            
            pdf_output.change(
                fn=download_markdown,
                inputs=[pdf_output],
                outputs=[pdf_download]
            )
        
        # Batch PDF Processing Tab
        with gr.Tab("Batch PDF Processing"):
            with gr.Row():
                with gr.Column():
                    batch_input = gr.File(
                        label="Upload PDFs",
                        file_types=[".pdf"],
                        file_count="multiple",
                        type="filepath"
                    )
                    batch_model = gr.Dropdown(
                        choices=available_models,
                        label="Select Model",
                        value=available_models[0] if available_models else None,
                        info="Choose the vision-language model to use"
                    )
                    
                    # Generation Parameters (Collapsible)
                    with gr.Accordion("Generation Parameters", open=False):
                        batch_temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.1,
                            step=0.1,
                            label="Temperature",
                            info="Controls randomness (0=deterministic, 1=creative)"
                        )
                        batch_top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top P",
                            info="Nucleus sampling threshold"
                        )
                        batch_max_tokens = gr.Slider(
                            minimum=128,
                            maximum=4096,
                            value=2048,
                            step=128,
                            label="Max Output Tokens",
                            info="Maximum length of generated text"
                        )
                    
                    batch_button = gr.Button("Process All PDFs", variant="primary")
                
                with gr.Column():
                    batch_status = gr.Textbox(
                        label="Processing Status",
                        lines=15,
                        interactive=False
                    )
                    batch_download = gr.File(label="Download Results (ZIP)")
            
            batch_button.click(
                fn=process_batch_pdfs,
                inputs=[batch_input, batch_model, batch_temperature, batch_top_p, batch_max_tokens],
                outputs=[batch_download, batch_status]
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
                    
                    # Generation Parameters (Collapsible)
                    with gr.Accordion("Generation Parameters", open=False):
                        image_temperature = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.1,
                            step=0.1,
                            label="Temperature",
                            info="Controls randomness (0=deterministic, 1=creative)"
                        )
                        image_top_p = gr.Slider(
                            minimum=0.0,
                            maximum=1.0,
                            value=0.9,
                            step=0.05,
                            label="Top P",
                            info="Nucleus sampling threshold"
                        )
                        image_max_tokens = gr.Slider(
                            minimum=128,
                            maximum=4096,
                            value=2048,
                            step=128,
                            label="Max Output Tokens",
                            info="Maximum length of generated text"
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
                inputs=[image_input, image_model, image_prompt, image_temperature, image_top_p, image_max_tokens],
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
                - **Batch PDF Processing**: Process multiple PDFs at once, download results as ZIP
                - **Image Processing**: Extract text from images directly
                - **Custom Prompts**: Provide specific instructions for extraction
                - **Markdown Output**: Results in clean, formatted markdown
                - **Download**: Save results as `.md` files or ZIP archives
                
                ## Tips
                
                - Larger models (32B, 72B) provide better accuracy but require more memory
                - Smaller models (3B, 7B) are faster and use less memory
                
                ## Custom Prompts
                
                You can provide custom prompts for specific extraction tasks:
                
                - "Extract all text preserving tables and formatting"
                - "Extract only the invoice number, date, and total amount"
                - "Extract the document in JSON format with sections"
                
                ## System Information
                
                ### Hardware
                - **CUDA Available**: {"✅ Yes" if system_info['cuda_available'] else "❌ No (CPU only)"}
                - **CUDA Version**: {system_info['cuda_version']}
                - **GPU Count**: {system_info['gpu_count']}
                
                {"### GPU Details" if system_info['gpu_count'] > 0 else ""}
                {chr(10).join([f"- **GPU {gpu['id']}**: {gpu['name']} ({gpu['memory']})" for gpu in system_info['gpus']]) if system_info['gpus'] else ""}
                
                ### Software
                - **Python**: {system_info['python_version']}
                - **PyTorch**: {system_info['pytorch_version']}
                - **Transformers**: {system_info['transformers_version']}
                - **Gradio**: {system_info['gradio_version']}
                - **Accelerate**: {system_info['accelerate_version']}
                - **Pillow**: {system_info['pillow_version']}
                
                ### Configuration
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

