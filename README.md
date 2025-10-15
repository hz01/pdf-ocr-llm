# PDF OCR with Large Language Models

A professional PDF OCR system that leverages state-of-the-art Large Language Models for high-quality text extraction. This project uses Qwen 2.5 VL vision-language models with built-in multi-GPU acceleration support.

## Features

- **🤖 State-of-the-art Models**: Qwen 2.5 VL vision-language models (3B, 7B, 32B, 72B)
- **🌐 Web UI**: User-friendly Gradio interface for easy interaction
- **🚀 REST API**: FastAPI backend for programmatic access
- **💻 CLI**: Command-line interface for automation
- **📝 Markdown Output**: Preserves document structure and formatting
- **🎯 Custom Prompts**: Flexible prompt engineering for specific tasks
- **⚡ Multi-GPU Support**: Automatic device distribution for large models
- **📦 Batch Processing**: Process multiple files efficiently
- **🔄 HuggingFace Integration**: Automatic model downloading and caching

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

# Or launch individually
python main.py ui        # Just the Web UI
python main.py api       # Just the API

# Or use CLI for direct processing
python main.py process document.pdf --model "Qwen2.5-VL-7B-Instruct" --output result.md
```

## Installation

### System Dependencies

Before installing Python dependencies, ensure you have the following system packages:

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
1. Download and install Poppler: https://github.com/oschwartz10612/poppler-windows/releases
2. Add to your system PATH

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
- 📄 Upload and process PDFs
- 🖼️ Upload and process images  
- 🎯 Select models from dropdown
- 📝 Custom prompts support
- ⬇️ Download results as markdown files
- 💡 Real-time status updates

### REST API Only (FastAPI)

Run just the API server:

```bash
python main.py api
```

API will be available at `http://localhost:8000`

**Options:**
- `--host HOST` - Server host (default: 0.0.0.0)
- `--port PORT` - Server port (default: 8000)

**API Endpoints:**

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
    │   ├── qwen2vl_model.py     # Qwen 2.5 VL implementation
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

### Qwen 2.5 VL

A state-of-the-art vision-language model from Alibaba Cloud. Available in three sizes:
- **2B parameters**: Faster processing, lower memory requirements (~4GB GPU)
- **7B parameters**: Balanced performance and quality (~14GB GPU)
- **72B parameters**: Highest quality output, requires significant resources (80GB+ GPU or multi-GPU)

## Multi-GPU Support

The project automatically distributes model layers across multiple GPUs when available. Configure in `config.yaml`:

```yaml
device:
  use_multi_gpu: true
  device_map: "auto"  # Automatic distribution
```
