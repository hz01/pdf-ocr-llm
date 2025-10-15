# PDF OCR with Large Language Models

A professional PDF OCR system that leverages state-of-the-art Large Language Models for high-quality text extraction. This project uses Qwen 2.5 VL vision-language models with built-in multi-GPU acceleration support.

## Features

- State-of-the-art Qwen 2.5 VL vision-language models:
  - 2B, 7B, and 72B parameter variants
- Markdown output format preserving document structure
- Multi-GPU support with automatic device distribution
- Professional class-based architecture
- Batch processing capabilities
- PDF to image conversion with configurable DPI
- Command-line interface for easy usage
- Configurable model selection via YAML
- Progress tracking and detailed logging

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

## Configuration

The project uses a YAML configuration file (`config.yaml`) to manage models and settings. You can customize:

- Available models and their parameters
- Inference settings (batch size, temperature, etc.)
- Device settings (GPU usage, multi-GPU)
- OCR settings (DPI, output format)

Example configuration structure:
```yaml
models:
  qwen2vl:
    - name: "Qwen2.5-VL-7B"
      model_id: "Qwen/Qwen2.5-VL-7B-Instruct"
      type: "qwen2vl"
      
device:
  use_multi_gpu: true
  device_map: "auto"
```

## Usage

### List Available Models

View all configured models:
```bash
python main.py list-models
```

### Process a Single PDF

Extract text from a PDF file to Markdown:
```bash
python main.py process input.pdf --model "Qwen2.5-VL-7B" --output output.md
```

Save intermediate images:
```bash
python main.py process input.pdf --model "Qwen2.5-VL-7B" --save-images --images-dir ./images
```

### Process a Single Image

Extract text from an image file to Markdown:
```bash
python main.py process image.png --model "Qwen2.5-VL-7B" --output output.md
```

### Batch Processing

Process multiple files at once:
```bash
python main.py batch --model "Qwen2.5-VL-7B" --input-dir ./pdfs --output-dir ./outputs
```

Process specific files:
```bash
python main.py batch --model "Qwen2.5-VL-7B" --input-files file1.pdf file2.pdf --output-dir ./outputs
```

### Custom Prompts

Use a custom prompt for better context:
```bash
python main.py process invoice.pdf --model "Qwen2.5-VL-7B" --prompt "Extract all text from this invoice, preserving table structure"
```

### Show Device Information

Display available GPUs:
```bash
python main.py process input.pdf --model "Qwen2.5-VL-7B" --show-device-info
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

## Performance Tips

1. **GPU Memory**: Use smaller models (2B, 7B) if running out of memory
2. **Batch Size**: Adjust in config.yaml based on available memory
3. **DPI**: Lower DPI (150-200) for faster processing, higher (300+) for better quality
4. **Multi-GPU**: Enable for large models (72B) or batch processing

## Troubleshooting

### CUDA Out of Memory

- Use a smaller model variant
- Reduce batch_size in config.yaml
- Enable multi-GPU mode
- Close other GPU-intensive applications

### PDF Conversion Fails

- Ensure poppler-utils is installed
- Check PDF file is not corrupted
- Try reducing DPI in config.yaml

### Model Download Issues

- Ensure you have internet connection
- Check Hugging Face Hub accessibility
- Models are cached in `~/.cache/huggingface/`

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Follow PEP 8 style guide
2. Add docstrings to all classes and methods
3. Include type hints
4. Write unit tests for new features
5. Update documentation

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Qwen team at Alibaba Cloud for the Qwen 2.5 VL models
- Hugging Face for the transformers library
- pdf2image contributors

## Citation

If you use this project in your research, please cite:

```bibtex
@software{pdf_ocr_llm,
  title={PDF OCR with Large Language Models},
  author={Your Name},
  year={2024},
  url={https://github.com/yourusername/pdf-ocr-llm}
}
```

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- Check existing issues for solutions
- Review documentation thoroughly

## Changelog

### Version 1.0.0
- Initial release
- Support for Qwen 2.5 VL (2B, 7B, 72B)
- Multi-GPU support
- Markdown output format
- Batch processing capabilities
- Command-line interface