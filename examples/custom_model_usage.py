"""
Example of using custom prompts and different models.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline.ocr_pipeline import OCRPipeline


def extract_invoice_data():
    """Example: Extract structured data from invoices."""
    pipeline = OCRPipeline("config.yaml")
    
    custom_prompt = """
    Extract the following information from this invoice:
    - Invoice number
    - Date
    - Total amount
    - Line items with descriptions and prices
    
    Format the output as structured text.
    """
    
    # Uncomment to use
    # result = pipeline.process_pdf(
    #     pdf_path="invoice.pdf",
    #     model_name="Qwen2.5-VL-7B",
    #     prompt=custom_prompt,
    #     output_path="invoice_data.txt"
    # )
    
    pipeline.cleanup()
    print("Invoice processing complete!")


def compare_models():
    """Example: Compare different models on the same document."""
    pipeline = OCRPipeline("config.yaml")
    
    document_path = "path/to/document.pdf"
    models_to_test = ["Qwen2.5-VL-2B", "Qwen2.5-VL-7B", "Qwen2.5-VL-72B"]
    
    results = {}
    
    # Uncomment to use
    # for model_name in models_to_test:
    #     print(f"\nTesting model: {model_name}")
    #     result = pipeline.process_pdf(
    #         pdf_path=document_path,
    #         model_name=model_name,
    #         output_path=f"output_{model_name}.txt"
    #     )
    #     results[model_name] = result
    #     print(f"Completed in {result.get('processing_time', 'N/A')} seconds")
    
    pipeline.cleanup()
    print("\nModel comparison complete!")


def process_with_custom_settings():
    """Example: Process with custom model settings."""
    from src.config.config_manager import ConfigManager
    from src.utils.device_manager import DeviceManager
    from src.models.model_factory import ModelFactory
    from PIL import Image
    
    config_manager = ConfigManager("config.yaml")
    device_manager = DeviceManager(config_manager.get_device_config())
    
    # Get model config
    model_config = config_manager.get_model_by_name("Qwen2.5-VL-7B")
    inference_config = config_manager.get_inference_config()
    
    # Create and load model
    model = ModelFactory.create_model(model_config, device_manager, inference_config)
    model.load_model()
    
    # Process with custom prompt
    # Uncomment to use
    # image = Image.open("document_page.png")
    # custom_prompt = "Extract all text and preserve the exact formatting including tables and lists."
    # result = model.process_image(image, prompt=custom_prompt)
    # 
    # print("Extracted text:", result)
    
    model.unload_model()
    print("Processing complete!")


if __name__ == "__main__":
    print("Custom Model Usage Examples")
    print("=" * 60)
    print("\nUncomment the function you want to test in the script.")
    print("\nAvailable examples:")
    print("1. extract_invoice_data() - Extract structured invoice data")
    print("2. compare_models() - Compare different model sizes")
    print("3. process_with_custom_settings() - Custom prompts and settings")

