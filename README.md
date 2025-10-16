# PDF OCR with Vision Language Models

A professional PDF OCR system that leverages state-of-the-art vision-language models for high-quality text extraction with built-in multi-GPU acceleration support.

## Features

- **State-of-the-art Models**: Multiple vision-language model families
  - Qwen 3 VL: 4B, 8B, 30B, 235B (Latest)
  - Qwen 2.5 VL: 3B, 7B, 32B, 72B
  - InternVL 3.5: 1B, 2B, 4B, 8B, 14B, 30B, 38B, 241B
- **Web UI**: User-friendly Gradio interface with batch processing
- **REST API**: FastAPI backend for programmatic access
- **CLI**: Command-line interface for automation
- **Batch Processing**: Process multiple PDFs with progress tracking and ZIP download
- **Markdown Output**: Clean formatting without code fences
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
python main.py process document.pdf --model "Qwen2.5-VL-7B-Instruct" --output result.md

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
  qwen25vl:
    - name: "Qwen2.5-VL-7B-Instruct"
      model_id: "Qwen/Qwen2.5-VL-7B-Instruct"
      type: "qwen25vl"
      
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
- Upload and process single PDFs
- Batch process multiple PDFs with ZIP download
- Upload and process images  
- Select models from dropdown
- Custom prompts support
- Download results as markdown files or ZIP archives
- Real-time progress tracking for batch processing
- Clean markdown output without code fences

### REST API Only (FastAPI)

Run just the API server:

```bash
python main.py api
```

API will be available at `http://localhost:8000`

**Options:**
- `--host HOST` - Server host (default: 0.0.0.0)
- `--port PORT` - Server port (default: 8000)

**Endpoints:**

- `GET /models` - List available models
- `POST /process/pdf` - Process a PDF file
- `POST /process/image` - Process an image file
- `GET /health` - Health check

**Example API Usage:**

```python
import requests

# List models
response = requests.get("http://localhost:8000/models")
print(response.json())

# Process PDF
with open("document.pdf", "rb") as f:
    files = {"file": f}
    data = {"model_name": "Qwen2.5-VL-7B-Instruct"}
    response = requests.post("http://localhost:8000/process/pdf", files=files, data=data)
    print(response.json()["markdown"])
```

**Interactive API Docs:** Visit `http://localhost:8000/docs` for Swagger UI

### Command Line Interface

#### List Available Models

View all configured models:
```bash
python main.py list-models
```

### Process a Single PDF

Extract text from a PDF file to Markdown:
```bash
python main.py process input.pdf --model "Qwen2.5-VL-7B-Instruct" --output output.md
```

Save intermediate images:
```bash
python main.py process input.pdf --model "Qwen2.5-VL-7B-Instruct" --save-images --images-dir ./images
```

### Process a Single Image

Extract text from an image file to Markdown:
```bash
python main.py process image.png --model "Qwen2.5-VL-7B-Instruct" --output output.md
```

### Batch Processing

Process multiple files at once:
```bash
python main.py batch --model "Qwen2.5-VL-7B-Instruct" --input-dir ./pdfs --output-dir ./outputs
```

Process specific files:
```bash
python main.py batch --model "Qwen2.5-VL-7B-Instruct" --input-files file1.pdf file2.pdf --output-dir ./outputs
```

### Custom Prompts

Use a custom prompt for better context:
```bash
python main.py process invoice.pdf --model "Qwen2.5-VL-7B-Instruct" --prompt "Extract all text from this invoice, preserving table structure"
```

### Show Device Information

Display available GPUs:
```bash
python main.py process input.pdf --model "Qwen2.5-VL-7B-Instruct" --show-device-info
```

## Project Structure

```
pdf-ocr-llm/
├── config.yaml              # Configuration file
├── main.py                  # Main entry point
├── requirements.txt         # Python dependencies
├── README.md               # This file
└── src/
    ├── config/
    │   └── config_manager.py    # Configuration management
    ├── models/
    │   ├── base_model.py        # Abstract base class
    │   ├── qwen25vl_model.py    # Qwen 2.5 VL implementation
    │   ├── internvl_model.py    # InternVL 3.5 implementation
    │   └── model_factory.py     # Model factory pattern
    ├── processors/
    │   └── pdf_processor.py     # PDF processing utilities
    ├── pipeline/
    │   └── ocr_pipeline.py      # Main OCR pipeline
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

The project supports multiple vision-language model families optimized for OCR and document understanding:

### Qwen 3 VL (Alibaba Cloud) - Latest

Next-generation vision-language models with cutting-edge OCR performance:

- **Qwen3-VL-4B-Instruct** - 4B parameters, highly efficient
- **Qwen3-VL-8B-Instruct** - 8B parameters, excellent balance
- **Qwen3-VL-30B-A3B-Instruct** - 30B parameters, very powerful
- **Qwen3-VL-235B-A22B-Instruct** - 235B parameters, state-of-the-art

> 📚 **Reference**: [Qwen 3 VL Collection on HuggingFace](https://huggingface.co/collections/Qwen/qwen3-vl-68d2a7c1b8a8afce4ebd2dbe)

### Qwen 2.5 VL (Alibaba Cloud)

Previous generation with proven reliability:

- **Qwen2.5-VL-3B-Instruct** - 3B parameters, efficient and fast
- **Qwen2.5-VL-7B-Instruct** - 7B parameters, balanced performance
- **Qwen2.5-VL-32B-Instruct** - 32B parameters, high quality
- **Qwen2.5-VL-72B-Instruct** - 72B parameters, best quality

### InternVL 3.5 (OpenGVLab)

Versatile multimodal models with dynamic image preprocessing and tiling:

- **InternVL3.5-1B-Instruct** - 1B parameters, ultra-lightweight
- **InternVL3.5-2B-Instruct** - 2B parameters, very efficient
- **InternVL3.5-4B-Instruct** - 4B parameters, balanced small model
- **InternVL3.5-8B-Instruct** - 8B parameters, strong mid-size
- **InternVL3.5-14B-Instruct** - 14B parameters, high performance
- **InternVL3.5-30B-A3B-Instruct** - 30B parameters, very powerful
- **InternVL3.5-38B-Instruct** - 38B parameters, flagship model
- **InternVL3.5-241B-A28B-Instruct** - 241B parameters, ultimate quality

> **Note**: InternVL models use dynamic image tiling for better quality on high-resolution documents.

**Model Selection Tips:**
- **For speed**: Use 1B-4B models (InternVL3.5-1B, Qwen3-VL-4B)
- **For quality**: Use 7B-14B models (Qwen3-VL-8B, InternVL3.5-8B)
- **For best results**: Use 30B+ models (Qwen3-VL-30B, Qwen3-VL-235B)
- **For high-resolution documents**: InternVL models (dynamic tiling)
- **For general OCR (recommended)**: Qwen 3 VL models (latest, best performance)

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
