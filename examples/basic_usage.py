"""
Basic usage example for PDF OCR with LLMs.

This script demonstrates how to use the OCR pipeline programmatically.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.ocr_pipeline import OCRPipeline


def main():
    """Example usage of the OCR pipeline."""
    
    # Initialize pipeline
    print("Initializing OCR Pipeline...")
    pipeline = OCRPipeline("config.yaml")
    
    # Show device information
    pipeline.device_manager.print_device_info()
    
    # List available models
    print("\nAvailable models:")
    models = pipeline.list_available_models()
    for model in models:
        print(f"  - {model['name']} ({model['type']})")
    
    # Example 1: Process a PDF
    print("\nExample 1: Processing a PDF file")
    print("-" * 60)
    
    # Uncomment and modify with your PDF path
    # result = pipeline.process_pdf(
    #     pdf_path="path/to/your/document.pdf",
    #     model_name="Qwen2.5-VL-7B",
    #     output_path="output.txt",
    #     save_images=True,
    #     images_dir="temp_images"
    # )
    # print(f"Processed {result['num_pages']} pages")
    # print(f"Output saved to: output.txt")
    
    # Example 2: Process a single image
    print("\nExample 2: Processing an image file")
    print("-" * 60)
    
    # Uncomment and modify with your image path
    # result = pipeline.process_image(
    #     image_path="path/to/your/image.png",
    #     model_name="Qwen2.5-VL-7B",
    #     output_path="image_output.txt"
    # )
    # print(f"Extracted text: {result['text'][:100]}...")
    
    # Example 3: Batch processing
    print("\nExample 3: Batch processing")
    print("-" * 60)
    
    # Uncomment and modify with your paths
    # input_files = [
    #     "path/to/doc1.pdf",
    #     "path/to/doc2.pdf",
    #     "path/to/image.png"
    # ]
    # results = pipeline.process_batch(
    #     input_paths=input_files,
    #     model_name="Qwen2.5-VL-7B",
    #     output_dir="batch_outputs"
    # )
    # print(f"Processed {len(results)} files")
    
    # Cleanup
    print("\nCleaning up...")
    pipeline.cleanup()
    print("Done!")


if __name__ == "__main__":
    main()

