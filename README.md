# PDF OCR with Vision Language Models

A professional PDF OCR system that leverages state-of-the-art vision-language models for high-quality text extraction with built-in multi-GPU acceleration support.

## Features

- **State-of-the-art Models**: Multiple vision-language model families
  - **Qwen 3 VL**: 4B, 8B, 30B, 235B (general-purpose OCR, markdown output)
  - **InternVL 3.5 Instruct**: 1B, 2B, 4B, 8B, 14B, 30B, 38B, 241B (dynamic tiling, markdown)
  - **OCRFlux-3B** (ChatDOC): Qwen2.5-VL based, document-to-markdown; output as structured JSON array per page
  - **GLM-OCR**: 0.9B (dedicated document OCR; 1 image per call; text/formula/table + JSON info extraction)
- **Web UI**: Gradio with **PDF Processing**, **Batch PDF**, **Image Processing**, and **Extract Info** tabs
- **REST API**: FastAPI backend for programmatic access
- **CLI**: Command-line interface for automation
- **Batch Processing**: Multiple PDFs with progress tracking and ZIP download
- **Extract Info**: Upload 1 PDF (all pages) or multiple images; GLM-OCR runs 1 image per call; compare view (preview left, JSON array right); strict JSON-schema prompts (e.g. ID cards, forms)
- **Output Formats**: `.md` (VLMs), `.txt` (GLM-OCR OCR), `.json` (OCRFlux full doc or Extract Info)
- **Per-model generation params**: Temperature/top_p/max tokens in UI adapt by model (e.g. GLM-OCR shows only max tokens)
- **Custom Prompts**: Flexible prompt engineering for specific tasks
- **Multi-GPU Support**: Automatic device distribution for large models
- **Cloud Ready**: Deploy on Kaggle, Colab, or HuggingFace Spaces
- **HuggingFace Integration**: Automatic model downloading and caching

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Set up HuggingFace token (optional)
cp env.example .env
# Edit .env and add your HF_TOKEN

# Launch both Web UI and API together (recommended)
python main.py serve
# Web UI:  http://localhost:7860
# API:     http://localhost:8000
# API Docs: http://localhost:8000/docs

# With public share link (for Kaggle, Colab, remote access)
python main.py serve --share

# Or launch individually
python main.py ui        # Just the Web UI
python main.py ui --share # With public link
python main.py api       # Just the API

# Or use CLI for direct processing
python main.py process document.pdf --model "Qwen3-VL-4B-Instruct" --output result.md

# List all available models
python main.py list-models
```

## Installation

### System Dependencies

Before installing Python dependencies, ensure you have Poppler installed:

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y poppler-utils
```

#### macOS
```bash
brew install poppler
```

#### Windows
1. Download Poppler: https://github.com/oschwartz10612/poppler-windows/releases
2. Extract and add to your system PATH

#### Kaggle/Colab
```bash
!apt-get install -y poppler-utils
```

### Python Dependencies

1. Clone the repository:
```bash
git clone https://github.com/yourusername/pdf-ocr-llm.git
cd pdf-ocr-llm
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install Python dependencies:
```bash
pip install -r requirements.txt
```

4. Set up environment variables (optional but recommended for HuggingFace authentication):
```bash
# Copy the example file
cp env.example .env

# Edit .env and add your HuggingFace token
# Get your token from: https://huggingface.co/settings/tokens
```

Your `.env` file should contain:
```
HF_TOKEN=your_huggingface_token_here
```

**Note:** The HuggingFace token is required if you're accessing gated models or private repositories. For public models, it's optional but recommended.

## Configuration

The project uses a YAML configuration file (`config.yaml`) to manage models and settings. You can customize:

- Available models and their parameters
- Inference settings (batch size, temperature, etc.)
- Device settings (GPU usage, multi-GPU)
- OCR settings (DPI, output format)

Example configuration structure:
```yaml
models:
  qwen3vl:
    - name: "Qwen3-VL-4B-Instruct"
      model_id: "Qwen/Qwen3-VL-4B-Instruct"
      type: "qwen3vl"

device:
  use_multi_gpu: true
  device_map: "auto"
```

## Usage

### Serve Both UI & API (Recommended)

Launch both the web interface and API server together:

```bash
python main.py serve
```

This starts:
- **Web UI** at `http://localhost:7860` - Interactive interface
- **REST API** at `http://localhost:8000` - Programmatic access
- **API Docs** at `http://localhost:8000/docs` - Interactive API documentation

**Options:**
- `--api-host HOST` - API server host (default: 0.0.0.0)
- `--api-port PORT` - API server port (default: 8000)
- `--ui-host HOST` - UI server host (default: 0.0.0.0)
- `--ui-port PORT` - UI server port (default: 7860)
- `--share` - Create public share link for UI (useful for remote access)

### Web UI Only (Gradio)

The easiest way to use the application is through the web interface:

```bash
python main.py ui
```

Then open your browser to `http://localhost:7860`

**Options:**
- `--host HOST` - Server host (default: 0.0.0.0)
- `--port PORT` - Server port (default: 7860)
- `--share` - Create public share link

**Features:**
- **PDF Processing**: Single PDF, custom prompt, download as `.md`, `.txt` (GLM-OCR), or `.json` (OCRFlux)
- **Batch PDF Processing**: Multiple PDFs, ZIP download
- **Image Processing**: Single image with same model and prompt options
- **Extract Info**: Upload **1 PDF** (all pages) or **multiple images**; GLM-OCR only, 1 image per call; paste a JSON-schema prompt (e.g. ID card, invoice); get a single JSON array (one object per page/image); **compare view**: preview gallery left, extracted JSON right; download as `.json`
- Select models from dropdown (Qwen3-VL, InternVL 3.5, OCRFlux-3B, GLM-OCR)
- **Generation parameters** in the first two tabs update per model (e.g. GLM-OCR shows only max tokens)
- Custom prompts support
- Download as `.md`, `.txt` (GLM-OCR), or `.json` (OCRFlux, Extract Info)
- Real-time progress tracking for batch processing

### REST API Only (FastAPI)

Run just the API server:

```bash
python main.py api
```

API will be available at `http://localhost:8000`

**Options:** `--host HOST` (default: 0.0.0.0), `--port PORT` (default: 8000)

#### Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | API info and list of endpoints |
| `GET` | `/models` | List available models (names and config) |
| `POST` | `/process/pdf` | Upload a PDF, get extracted text |
| `POST` | `/process/image` | Upload an image, get extracted text |
| `GET` | `/health` | Health check |

#### Process PDF — `POST /process/pdf`

- **Request:** `multipart/form-data`
  - `file` (required): PDF file
  - `model_name` (required): Exact model name from `/models` (e.g. `Qwen3-VL-4B-Instruct`, `GLM-OCR`, `InternVL3.5-8B-Instruct`)
  - `prompt` (optional): Custom OCR prompt; omit for default
- **Response (200):** JSON
  - `success`: `true`
  - `model`: model name used
  - `num_pages`: number of pages
  - `markdown`: extracted text (markdown for VLMs, plain text for GLM-OCR, JSON string for OCRFlux)
  - `processing_time`: (if present) seconds

#### Process image — `POST /process/image`

- **Request:** `multipart/form-data`
  - `file` (required): Image file (e.g. PNG, JPEG)
  - `model_name` (required): Model name from `/models`
  - `prompt` (optional): Custom prompt
- **Response (200):** JSON
  - `success`, `model`, `markdown`, `text_length`

#### Example: cURL

```bash
# List models
curl http://localhost:8000/models

# Process a PDF
curl -X POST http://localhost:8000/process/pdf \
  -F "file=@document.pdf" \
  -F "model_name=Qwen3-VL-4B-Instruct"

# With custom prompt
curl -X POST http://localhost:8000/process/pdf \
  -F "file=@document.pdf" \
  -F "model_name=GLM-OCR" \
  -F "prompt=Text Recognition:"

# Process an image
curl -X POST http://localhost:8000/process/image \
  -F "file=@page.png" \
  -F "model_name=InternVL3.5-8B-Instruct"
```

#### Example: Python

```python
import requests

BASE = "http://localhost:8000"

# List models
r = requests.get(f"{BASE}/models")
print(r.json())  # {"models": [{"name": "Qwen3-VL-4B-Instruct", ...}, ...]}

# Process PDF
with open("document.pdf", "rb") as f:
    r = requests.post(
        f"{BASE}/process/pdf",
        files={"file": ("document.pdf", f, "application/pdf")},
        data={"model_name": "Qwen3-VL-4B-Instruct"}
    )
data = r.json()
print(data["markdown"])   # extracted text
print(data["num_pages"])

# Process image with optional prompt
with open("scan.png", "rb") as f:
    r = requests.post(
        f"{BASE}/process/image",
        files={"file": ("scan.png", f, "image/png")},
        data={
            "model_name": "GLM-OCR",
            "prompt": "Text Recognition:"  # or "Formula Recognition:" / "Table Recognition:"
        }
    )
print(r.json()["markdown"])
```

**Interactive API docs:** Open `http://localhost:8000/docs` (Swagger UI) to try the endpoints from the browser.

### Command Line Interface

#### List Available Models

View all configured models:
```bash
python main.py list-models
```

### Process a Single PDF

Extract text from a PDF file to Markdown:
```bash
python main.py process input.pdf --model "Qwen3-VL-4B-Instruct" --output output.md
```

Save intermediate images:
```bash
python main.py process input.pdf --model "Qwen3-VL-4B-Instruct" --save-images --images-dir ./images
```

### Process a Single Image

Extract text from an image file to Markdown:
```bash
python main.py process image.png --model "Qwen3-VL-4B-Instruct" --output output.md
```

### Batch Processing

Process multiple files at once:
```bash
python main.py batch --model "Qwen3-VL-4B-Instruct" --input-dir ./pdfs --output-dir ./outputs
```

Process specific files:
```bash
python main.py batch --model "Qwen3-VL-4B-Instruct" --input-files file1.pdf file2.pdf --output-dir ./outputs
```

### Custom Prompts

Use a custom prompt for better context:
```bash
python main.py process invoice.pdf --model "Qwen3-VL-4B-Instruct" --prompt "Extract all text from this invoice, preserving table structure"
```

### Show Device Information

Display available GPUs:
```bash
python main.py process input.pdf --model "Qwen3-VL-4B-Instruct" --show-device-info
```

## Project Structure

```
pdf-ocr-llm/
├── config.yaml              # Configuration file
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
├── README.md                # This file
└── src/
    ├── config/
    │   └── config_manager.py    # Configuration management
    ├── models/
    │   ├── base_model.py        # Abstract base class
    │   ├── qwen3vl_model.py     # Qwen 3 VL implementation
    │   ├── internvl_model.py    # InternVL 3.5 Instruct implementation
    │   ├── ocrflux_model.py     # OCRFlux-3B (ChatDOC, markdown/JSON)
    │   ├── glm_ocr_model.py     # GLM-OCR (1 image per call; document + info extraction)
    │   └── model_factory.py     # Model factory pattern
    ├── processors/
    │   └── pdf_processor.py     # PDF processing utilities
    ├── pipeline/
    │   └── ocr_pipeline.py      # Main OCR pipeline
    ├── ui/
    │   └── app.py               # Gradio Web UI
    ├── api/
    │   └── app.py               # FastAPI REST API
    └── utils/
        └── device_manager.py    # Device management (GPU)
```

## Architecture

The project follows a professional, modular architecture:

### Core Components

1. **ConfigManager**: Manages configuration loading and validation
2. **DeviceManager**: Handles multi-GPU device allocation
3. **PDFProcessor**: Converts PDFs to images and handles image I/O
4. **BaseOCRModel**: Abstract base class for all OCR models
5. **ModelFactory**: Creates model instances based on configuration
6. **OCRPipeline**: Coordinates the entire OCR workflow

### Design Patterns

- Factory Pattern for model creation
- Strategy Pattern for different OCR models
- Dependency Injection for configuration and device management
- Template Method for common OCR operations

## Models

The project supports multiple vision-language model families for OCR and document understanding:

### Qwen 3 VL (Alibaba Cloud)

General-purpose vision-language models with strong OCR and markdown output:

- **Qwen3-VL-4B-Instruct** - 4B parameters, highly efficient
- **Qwen3-VL-8B-Instruct** - 8B parameters, excellent balance
- **Qwen3-VL-30B-A3B-Instruct** - 30B parameters, very powerful
- **Qwen3-VL-235B-A22B-Instruct** - 235B parameters, state-of-the-art

> 📚 **Reference**: [Qwen 3 VL on HuggingFace](https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe)

### InternVL 3.5 Instruct (OpenGVLab)

Versatile multimodal models with dynamic image preprocessing and tiling; output is markdown:

- **InternVL3.5-1B-Instruct** - 1B parameters, ultra-lightweight
- **InternVL3.5-2B-Instruct** - 2B parameters, very efficient
- **InternVL3.5-4B-Instruct** - 4B parameters, balanced
- **InternVL3.5-8B-Instruct** - 8B parameters, strong mid-size
- **InternVL3.5-14B-Instruct** - 14B parameters, high performance
- **InternVL3.5-30B-A3B-Instruct** - 30B parameters (31B), very powerful
- **InternVL3.5-38B-Instruct** - 38B parameters, flagship
- **InternVL3.5-241B-A28B-Instruct** - 241B parameters, maximum quality

> **Note**: InternVL uses the GitHub/Instruct format (`trust_remote_code`) and dynamic tiling for high-resolution documents.  
> 📚 **Reference**: [InternVL3.5-1B-Instruct](https://huggingface.co/OpenGVLab/InternVL3_5-1B-Instruct)

### GLM-OCR (Z.ai)

Dedicated document OCR model (0.9B); not a general-purpose VLM. Output is plain text by default. Supports:

- **Document parsing**: `"Text Recognition:"`, `"Formula Recognition:"`, `"Table Recognition:"`
- **Information extraction**: Prompts must follow a strict JSON schema (e.g. ID card fields, invoice fields). Use the **Extract Info from PDF** tab in the UI.

When using GLM-OCR for standard OCR, downloads use `.txt`; for **Extract Info**, output is a JSON array (one object per image). GLM-OCR accepts **one image per call**; the app batches by running it once per PDF page or per uploaded image.

> 📚 **Reference**: [GLM-OCR on HuggingFace](https://huggingface.co/zai-org/GLM-OCR)

### OCRFlux-3B (ChatDOC)

Document OCR model (3B, Qwen2.5-VL based); outputs clean markdown. The pipeline returns a **JSON array** (one object per page with fields such as `natural_text`, `primary_language`, `is_table`). Download uses `.json`.

> 📚 **Reference**: [OCRFlux-3B on HuggingFace](https://huggingface.co/ChatDOC/OCRFlux-3B)

**Note:** Qwen 2.5 VL (3B, 7B, 32B, 72B) has been removed in favor of Qwen 3 VL and the current model set. Use `config.yaml` to adjust which models are available.

### Model selection tips

- **Speed**: InternVL3.5-1B, Qwen3-VL-4B, OCRFlux-3B, or GLM-OCR
- **Quality (general OCR)**: Qwen3-VL-8B or InternVL3.5-8B
- **Document-to-markdown (JSON per page)**: OCRFlux-3B
- **Best quality**: Qwen3-VL-30B / 235B or InternVL3.5-38B / 241B
- **High-resolution / tiling**: InternVL 3.5
- **Structured extraction (forms, ID cards)**: GLM-OCR in **Extract Info** (1 PDF or multiple images; 1 image per call)

## Multi-GPU Support

The project automatically distributes model layers across multiple GPUs when available. Configure in `config.yaml`:

```yaml
device:
  use_multi_gpu: true
  device_map: "auto"  # Automatic distribution
```

## Deployment Options

### Local Development

Run locally with all features:
```bash
python main.py serve
```

### Cloud Notebooks (Kaggle, Colab)

Perfect for free GPU access with public sharing:

**Kaggle Notebook:**
```python
# Clone and setup
!git clone https://github.com/yourusername/pdf-ocr-llm.git
%cd pdf-ocr-llm
!pip install -r requirements.txt

# Launch with public link
!python main.py ui --share
# Or both API + UI
!python main.py serve --share
```

**Google Colab:**
```python
# Same as Kaggle
!git clone https://github.com/yourusername/pdf-ocr-llm.git
%cd pdf-ocr-llm
!pip install -r requirements.txt

# Launch with sharing enabled
!python main.py ui --share
```

The `--share` flag creates a temporary public URL (valid for 72 hours) that you can access from anywhere.

### Docker Deployment

Using Docker Compose:
```bash
docker-compose up
```

Access at:
- UI: `http://localhost:7860`
- API: `http://localhost:8000`

### Production Server

For production deployment with Nginx/Apache reverse proxy:
```bash
# Run API and UI on different ports
python main.py api --host 0.0.0.0 --port 8000
python main.py ui --host 0.0.0.0 --port 7860

# Or use serve for both
python main.py serve --api-port 8000 --ui-port 7860
```
