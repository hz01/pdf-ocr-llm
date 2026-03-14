"""
Gradio Web UI for PDF OCR with LLMs.

This module provides a user-friendly web interface for processing PDFs and images
using Qwen 2.5 VL models.
"""

import gradio as gr
import tempfile
import os
import time
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

# Models that support information extraction (GLM-OCR with JSON schema prompts)
extract_info_models = [
    m["name"] for m in pipeline.list_available_models()
    if m.get("type") == "glm_ocr"
]

# Default extraction prompt for GLM-OCR information extraction (strict JSON schema)
DEFAULT_EXTRACT_INFO_PROMPT = """{
    "id_number": "",
    "last_name": "",
    "first_name": "",
    "date_of_birth": "",
    "address": {
        "street": "",
        "city": "",
        "state": "",
        "zip_code": ""
    },
    "dates": {
        "issue_date": "",
        "expiration_date": ""
    },
    "sex": ""
}"""

# Resize threshold from config (for checkbox label)
max_image_size_px = pipeline.config_manager.get_ocr_config().get('max_image_size', 1536)

# Per-model generation parameters (temperature, top_p, max_new_tokens, max_new_tokens_slider_max)
# Used to update the Generation Parameters accordion when the model dropdown changes.
MODEL_GENERATION_PARAMS = {
    "qwen3vl": {"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 2048, "max_slider": 4096},
    "internvl": {"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 2048, "max_slider": 4096},
    "glm_ocr": {"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 8192, "max_slider": 8192, "show_temp_top_p": False},
}
DEFAULT_GENERATION_PARAMS = {"temperature": 0.1, "top_p": 0.9, "max_new_tokens": 2048, "max_slider": 4096, "show_temp_top_p": True}


def get_generation_param_updates(model_name):
    """Return gr.update() for temperature, top_p, max_tokens so the accordion reflects the selected model."""
    if not model_name:
        cfg = DEFAULT_GENERATION_PARAMS
        show_temp = True
    else:
        m = pipeline.config_manager.get_model_by_name(model_name)
        type_key = (m or {}).get("type", "")
        cfg = MODEL_GENERATION_PARAMS.get(type_key, DEFAULT_GENERATION_PARAMS)
        show_temp = cfg.get("show_temp_top_p", True)
    return (
        gr.update(value=cfg["temperature"], visible=show_temp),
        gr.update(value=cfg["top_p"], visible=show_temp),
        gr.update(value=cfg["max_new_tokens"], maximum=cfg["max_slider"], visible=True),
    )


def output_extension_for_model(model_name):
    """Use .txt for GLM-OCR (dedicated OCR); .md for general-purpose VL models."""
    if not model_name:
        return ".md"
    cfg = pipeline.config_manager.get_model_by_name(model_name)
    return ".txt" if (cfg and cfg.get("type") == "glm_ocr") else ".md"

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

    # Flash Attention (used by some models for faster inference)
    try:
        import flash_attn
        info['flash_attn_installed'] = True
        info['flash_attn_version'] = getattr(flash_attn, '__version__', 'unknown')
    except ImportError:
        info['flash_attn_installed'] = False
        info['flash_attn_version'] = None

    return info

system_info = get_system_info()


def process_pdf(pdf_file, model_name, custom_prompt, temperature, top_p, max_tokens, resize_high_res):
    """
    Process a PDF file and extract text.
    
    Args:
        pdf_file: Uploaded PDF file path
        model_name: Selected model name
        custom_prompt: Custom prompt (optional)
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Maximum output tokens
        resize_high_res: If True, resize images above threshold (faster)
    
    Returns:
        Extracted markdown text
    """
    if pdf_file is None:
        return "Please upload a PDF file."
    
    if not model_name:
        return "Please select a model."
    
    try:
        ext = output_extension_for_model(model_name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext, mode='w', encoding='utf-8') as temp_output:
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
            max_new_tokens=int(max_tokens),
            resize_high_res=resize_high_res
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


def process_image(image_file, model_name, custom_prompt, temperature, top_p, max_tokens, resize_high_res):
    """
    Process an image file and extract text.
    
    Args:
        image_file: Uploaded image file path
        model_name: Selected model name
        custom_prompt: Custom prompt (optional)
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Maximum output tokens
        resize_high_res: If True, resize images above threshold (faster)
    
    Returns:
        Extracted markdown text
    """
    if image_file is None:
        return "Please upload an image file."
    
    if not model_name:
        return "Please select a model."
    
    try:
        ext = output_extension_for_model(model_name)
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext, mode='w', encoding='utf-8') as temp_output:
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
            max_new_tokens=int(max_tokens),
            resize_high_res=resize_high_res
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


def download_result(text, model_name=None):
    """Create a downloadable file. Uses .txt for GLM-OCR, .md for other models."""
    if not text or text.startswith("Error") or text.startswith("Please"):
        return None
    ext = output_extension_for_model(model_name)
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext, mode='w', encoding='utf-8') as f:
        f.write(text)
        return f.name


def process_batch_pdfs(pdf_files, model_name, temperature, top_p, max_tokens, resize_high_res, progress=gr.Progress()):
    """
    Process multiple PDF files and create a zip with results.
    
    Args:
        pdf_files: List of uploaded PDF files
        model_name: Selected model name
        temperature: Sampling temperature
        top_p: Top-p sampling
        max_tokens: Maximum output tokens
        resize_high_res: If True, resize images above threshold (faster)
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
        ext = output_extension_for_model(model_name)
        for idx, pdf_file in enumerate(pdf_files, 1):
            # Update progress
            progress(idx / total_files, desc=f"Processing {idx}/{total_files}: {Path(pdf_file.name).name}")
            
            try:
                # Get original filename without extension
                pdf_name = Path(pdf_file.name).stem
                output_path = os.path.join(output_dir, f"{pdf_name}{ext}")
                
                # Process PDF
                result = pipeline.process_pdf(
                    pdf_path=pdf_file.name,
                    model_name=model_name,
                    output_path=output_path,
                    temperature=temperature,
                    top_p=top_p,
                    max_new_tokens=int(max_tokens),
                    resize_high_res=resize_high_res
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
            for out_file in Path(output_dir).glob(f"*{ext}"):
                zipf.write(out_file, out_file.name)
        
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


def clean_extract_info_output(text):
    """
    Strip non-JSON cruft from GLM-OCR extract-info output:
    '# Page N' headers, ```json/``` fences, trailing <|user|>. Unwrap first, then strip rest.
    """
    import re
    if not text or not text.strip():
        return text
    text = text.strip()
    # Strip leading "# Page N" lines before unwrap
    text = re.sub(r"^#\s*Page\s+\d+\s*\n*", "", text, flags=re.IGNORECASE)
    text = text.strip()
    # Unwrap first (so we don't strip closing ``` before matching).
    # Allow closing ``` to be followed by optional <|user|> so we still match.
    def unwrap_code_blocks(s):
        out = []
        # Match ```json?\n? content ``` optional <|user|>
        for m in re.finditer(
            r"`{3,}(?:json)?\s*\n?(.*?)`{3,}(?:\s*<\|[^|]+\|>)?\s*", s, re.DOTALL
        ):
            out.append(m.group(1).strip())
        if out:
            return "\n\n".join(out)
        return s
    text = unwrap_code_blocks(text)
    text = text.strip()
    # Strip any remaining trailing/leading cruft (backticks, tokens)
    text = re.sub(r"<\|[^|]+\|>\s*$", "", text)
    text = re.sub(r"^`+(?:json)?\s*\n?", "", text)
    text = re.sub(r"`+\s*$", "", text)
    return text.strip()


def process_pdf_extract_info(pdf_file, model_name, extraction_prompt, resize_high_res):
    """
    Process a PDF with GLM-OCR in information-extraction mode (JSON schema prompt).
    Returns extracted structured info as text (JSON per page).
    """
    if pdf_file is None:
        return "", "Please upload a PDF file."
    if not model_name:
        return "", "Please select a model (GLM-OCR)."
    if not extraction_prompt or not extraction_prompt.strip():
        return "", "Please provide an extraction prompt (JSON schema instruction)."
    try:
        start = time.perf_counter()
        ext = ".json"
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext, mode="w", encoding="utf-8") as temp_output:
            temp_output_path = temp_output.name

        result = pipeline.process_pdf(
            pdf_path=pdf_file,
            model_name=model_name,
            output_path=temp_output_path,
            prompt=extraction_prompt.strip(),
            temperature=0.1,
            top_p=0.9,
            max_new_tokens=8192,
            resize_high_res=resize_high_res,
        )

        with open(temp_output_path, "r", encoding="utf-8") as f:
            output_text = f.read()
        os.unlink(temp_output_path)

        elapsed = time.perf_counter() - start
        status = (
            f"Extracted info from {result.get('num_pages', 0)} page(s) in {elapsed:.2f} s"
        )
        return output_text, status
    except Exception as e:
        logger.error(f"Extract info failed: {e}")
        return "", f"Extraction failed: {str(e)}"


def download_extract_result(text):
    """Create a downloadable JSON file for extract-info tab."""
    if not text or text.startswith("Please") or text.startswith("Error") or text.startswith("Extraction failed"):
        return None
    with tempfile.NamedTemporaryFile(delete=False, suffix=".json", mode="w", encoding="utf-8") as f:
        f.write(text)
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
                    
                    pdf_resize_high_res = gr.Checkbox(
                        value=True,
                        label="Resize high-resolution images (faster)",
                        info=f"If checked, images with longest side > {max_image_size_px}px are resized (keeps aspect ratio). Uncheck to use original resolution."
                    )
                    pdf_button = gr.Button("Process PDF", variant="primary")
                
                with gr.Column():
                    pdf_status = gr.Textbox(label="Status", interactive=False)
                    pdf_output = gr.Textbox(
                        label="Extracted Markdown",
                        lines=20
                    )
                    pdf_download = gr.File(label="Download Result")
            
            pdf_button.click(
                fn=process_pdf,
                inputs=[pdf_input, pdf_model, pdf_prompt, pdf_temperature, pdf_top_p, pdf_max_tokens, pdf_resize_high_res],
                outputs=[pdf_output, pdf_status]
            )
            pdf_model.change(
                fn=get_generation_param_updates,
                inputs=[pdf_model],
                outputs=[pdf_temperature, pdf_top_p, pdf_max_tokens],
            )
            pdf_output.change(
                fn=download_result,
                inputs=[pdf_output, pdf_model],
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
                    
                    batch_resize_high_res = gr.Checkbox(
                        value=True,
                        label="Resize high-resolution images (faster)",
                        info=f"If checked, images with longest side > {max_image_size_px}px are resized. Uncheck to use original resolution."
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
                inputs=[batch_input, batch_model, batch_temperature, batch_top_p, batch_max_tokens, batch_resize_high_res],
                outputs=[batch_download, batch_status]
            )
            batch_model.change(
                fn=get_generation_param_updates,
                inputs=[batch_model],
                outputs=[batch_temperature, batch_top_p, batch_max_tokens],
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
                    
                    image_resize_high_res = gr.Checkbox(
                        value=True,
                        label="Resize high-resolution images (faster)",
                        info=f"If checked, images with longest side > {max_image_size_px}px are resized (keeps aspect ratio). Uncheck to use original resolution."
                    )
                    image_button = gr.Button("Process Image", variant="primary")
                
                with gr.Column():
                    image_status = gr.Textbox(label="Status", interactive=False)
                    image_output = gr.Textbox(
                        label="Extracted Markdown",
                        lines=20
                    )
                    image_download = gr.File(label="Download Result")
            
            image_button.click(
                fn=process_image,
                inputs=[image_input, image_model, image_prompt, image_temperature, image_top_p, image_max_tokens, image_resize_high_res],
                outputs=[image_output, image_status]
            )
            
            image_output.change(
                fn=download_result,
                inputs=[image_output, image_model],
                outputs=[image_download]
            )

        # Extract Info from PDF (GLM-OCR, information extraction with JSON schema)
        with gr.Tab("Extract Info from PDF"):
            with gr.Row():
                with gr.Column():
                    extract_pdf_input = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    extract_model = gr.Dropdown(
                        choices=extract_info_models,
                        value=extract_info_models[0] if extract_info_models else None,
                        label="Model",
                        info="GLM-OCR only (information extraction mode)",
                        interactive=len(extract_info_models) > 1,
                    )
                    extract_prompt = gr.Textbox(
                        label="Extraction prompt (JSON schema)",
                        placeholder="Use the default or paste your JSON schema instruction...",
                        value=DEFAULT_EXTRACT_INFO_PROMPT,
                        lines=22,
                        info="Strict JSON schema.",
                    )
                    extract_resize = gr.Checkbox(
                        value=True,
                        label="Resize high-resolution images (faster)",
                        info=f"If checked, images with longest side > {max_image_size_px}px are resized.",
                    )
                    extract_button = gr.Button("Extract Info", variant="primary")
                with gr.Column():
                    extract_status = gr.Textbox(label="Status", interactive=False)
                    extract_output = gr.Textbox(
                        label="Extracted structured info (JSON)",
                        lines=20,
                    )
                    extract_download = gr.File(label="Download as JSON")
            extract_button.click(
                fn=process_pdf_extract_info,
                inputs=[extract_pdf_input, extract_model, extract_prompt, extract_resize],
                outputs=[extract_output, extract_status],
            )
            extract_output.change(
                fn=download_extract_result,
                inputs=[extract_output],
                outputs=[extract_download],
            )
        
        # Information Tab
        with gr.Tab("Information"):
            # Build system info markdown dynamically
            gpu_details = ""
            if system_info['gpu_count'] > 0:
                gpu_details = "\n### GPU Details\n"
                for gpu in system_info['gpus']:
                    gpu_details += f"- **GPU {gpu['id']}**: {gpu['name']} ({gpu['memory']})\n"
            
            models_list = "\n".join([f"- **{model}**" for model in available_models])
            
            info_markdown = f"""
## Available Models

{models_list}

## Features

- **PDF Processing**: Convert PDF pages to images and extract text
- **Batch PDF Processing**: Process multiple PDFs at once, download results as ZIP
- **Image Processing**: Extract text from images directly
- **Extract Info from PDF**: GLM-OCR information extraction with a strict JSON schema (e.g. ID cards, forms)
- **Custom Prompts**: Provide specific instructions for extraction
- **Markdown Output**: Results in clean, formatted markdown (or plain text for GLM-OCR)
- **Download**: Save results as `.md`, `.txt` (GLM-OCR), `.json` (extract info), or ZIP archives

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
{gpu_details}
### Software
- **Python**: {system_info['python_version']}
- **PyTorch**: {system_info['pytorch_version']}
- **Transformers**: {system_info['transformers_version']}
- **Gradio**: {system_info['gradio_version']}
- **Accelerate**: {system_info['accelerate_version']}
- **Pillow**: {system_info['pillow_version']}
- **Flash Attention**: {"✅ Yes" if system_info['flash_attn_installed'] else "❌ No"}{f" (v{system_info['flash_attn_version']})" if system_info.get('flash_attn_installed') and system_info.get('flash_attn_version') else ""}

### Configuration
- **Pipeline Status**: Ready
- **Available Models**: {len(available_models)}
- **Config File**: config.yaml
"""
            
            gr.Markdown(info_markdown)

    # Apply per-model generation params on load (first model in list)
    def apply_initial_generation_params():
        u = get_generation_param_updates(available_models[0] if available_models else None)
        return list(u) + list(u)

    demo.load(
        fn=apply_initial_generation_params,
        outputs=[pdf_temperature, pdf_top_p, pdf_max_tokens, batch_temperature, batch_top_p, batch_max_tokens],
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

