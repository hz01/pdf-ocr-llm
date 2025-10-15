# Contributing to PDF OCR LLM

Thank you for your interest in contributing to this project. This document provides guidelines for contributing.

## Code of Conduct

Be respectful and constructive in all interactions.

## How to Contribute

### Reporting Bugs

Before creating a bug report:
1. Check existing issues to avoid duplicates
2. Gather information about the bug
3. Create a detailed bug report

Bug reports should include:
- Clear, descriptive title
- Steps to reproduce
- Expected vs actual behavior
- System information (OS, Python version, GPU)
- Error messages and logs
- Sample files (if applicable)

### Suggesting Enhancements

Enhancement suggestions should include:
- Clear description of the feature
- Use cases and benefits
- Potential implementation approach
- Any drawbacks or considerations

### Pull Requests

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Run tests and linters
5. Commit with clear messages
6. Push to your fork
7. Open a pull request

## Development Setup

### Clone and Setup

```bash
git clone https://github.com/yourusername/pdf-ocr-llm.git
cd pdf-ocr-llm
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Code Style

We follow PEP 8 style guidelines:

- Use 4 spaces for indentation
- Maximum line length: 100 characters
- Use descriptive variable names
- Add docstrings to all classes and methods
- Include type hints

Example:
```python
def process_document(
    document_path: str,
    model_name: str,
    output_dir: Optional[str] = None
) -> Dict[str, Any]:
    """
    Process a document using the specified model.
    
    Args:
        document_path: Path to the document file
        model_name: Name of the model to use
        output_dir: Optional directory for output files
        
    Returns:
        Dictionary containing processing results
    """
    pass
```

### Documentation

- Add docstrings to all public classes, methods, and functions
- Update README.md if adding new features
- Add examples for new functionality
- Keep CHANGELOG.md updated

### Testing

Before submitting:
1. Test your changes thoroughly
2. Ensure existing functionality still works
3. Add tests for new features
4. Run linters

## Project Structure

```
src/
├── config/          # Configuration management
├── models/          # Model implementations
├── processors/      # PDF and image processing
├── pipeline/        # Main OCR pipeline
└── utils/           # Utility functions
```

## Adding a New Model

To add support for a new OCR model:

1. Create a new model class in `src/models/`
2. Inherit from `BaseOCRModel`
3. Implement required methods:
   - `load_model()`
   - `process_image()`
   - `process_batch()`
4. Update `ModelFactory` in `src/models/model_factory.py`
5. Add model configuration to `config.yaml`
6. Add tests and documentation

Example:
```python
from .base_model import BaseOCRModel
from typing import List, Dict, Any
from PIL import Image

class NewModel(BaseOCRModel):
    def load_model(self) -> None:
        # Implementation
        pass
    
    def process_image(self, image: Image.Image, prompt: str = None) -> str:
        # Implementation
        pass
    
    def process_batch(self, images: List[Image.Image], prompts: List[str] = None) -> List[str]:
        # Implementation
        pass
```

## Commit Messages

Follow conventional commit format:

- `feat:` New feature
- `fix:` Bug fix
- `docs:` Documentation changes
- `style:` Code style changes (formatting)
- `refactor:` Code refactoring
- `test:` Adding or updating tests
- `chore:` Maintenance tasks

Examples:
```
feat: add support for BERT-based OCR model
fix: resolve GPU memory leak in batch processing
docs: update installation instructions for Windows
```

## Review Process

1. All submissions require review
2. Maintainers will provide feedback
3. Address review comments
4. Once approved, changes will be merged

## Questions?

Feel free to open an issue for any questions about contributing.

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

