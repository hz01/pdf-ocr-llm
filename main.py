import argparse
import logging
from pathlib import Path
import sys

from src.pipeline.ocr_pipeline import OCRPipeline


def setup_logging(log_level: str = "INFO") -> None:
    """
    Setup logging configuration.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


def list_models(args) -> None:
    """List all available models."""
    pipeline = OCRPipeline(args.config)
    models = pipeline.list_available_models()
    
    print("\n" + "=" * 80)
    print("Available Models:")
    print("=" * 80)
    
    for model in models:
        print(f"\nName: {model['name']}")
        print(f"  Model ID: {model['model_id']}")
        print(f"  Type: {model['type']}")
    
    print("\n" + "=" * 80)


def process_file(args) -> None:
    """Process a single PDF or image file."""
    pipeline = OCRPipeline(args.config)
    
    # Show device information
    if args.show_device_info:
        pipeline.device_manager.print_device_info()
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    # Determine output path
    output_path = None
    if args.output:
        output_path = Path(args.output)
    elif args.output_dir:
        output_dir = Path(args.output_dir)
        output_path = output_dir / f"{input_path.stem}_ocr.md"
    
    # Process based on file type
    if input_path.suffix.lower() == '.pdf':
        result = pipeline.process_pdf(
            input_path,
            model_name=args.model,
            output_path=output_path,
            save_images=args.save_images,
            images_dir=args.images_dir,
            prompt=args.prompt
        )
        
        print("\n" + "=" * 80)
        print(f"PDF Processing Complete")
        print("=" * 80)
        print(f"PDF: {result['pdf_path']}")
        print(f"Model: {result['model_name']}")
        print(f"Pages processed: {result['num_pages']}")
        
        if output_path:
            print(f"Output saved to: {output_path}")
        else:
            print("\n" + "-" * 80)
            print("Extracted Text:")
            print("-" * 80)
            print(result['full_text'])
        
        print("=" * 80 + "\n")
    else:
        result = pipeline.process_image(
            input_path,
            model_name=args.model,
            output_path=output_path,
            prompt=args.prompt
        )
        
        print("\n" + "=" * 80)
        print(f"Image Processing Complete")
        print("=" * 80)
        print(f"Image: {result['image_path']}")
        print(f"Model: {result['model_name']}")
        
        if output_path:
            print(f"Output saved to: {output_path}")
        else:
            print("\n" + "-" * 80)
            print("Extracted Text:")
            print("-" * 80)
            print(result['text'])
        
        print("=" * 80 + "\n")
    
    # Cleanup
    pipeline.cleanup()


def process_batch(args) -> None:
    """Process multiple files in batch."""
    pipeline = OCRPipeline(args.config)
    
    # Show device information
    if args.show_device_info:
        pipeline.device_manager.print_device_info()
    
    # Collect input files
    input_paths = []
    
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            print(f"Error: Input directory not found: {input_dir}")
            sys.exit(1)
        
        # Collect PDF and image files
        for pattern in ['*.pdf', '*.png', '*.jpg', '*.jpeg', '*.tiff', '*.bmp']:
            input_paths.extend(input_dir.glob(pattern))
    elif args.input_files:
        input_paths = [Path(f) for f in args.input_files]
    else:
        print("Error: Either --input-dir or --input-files must be specified for batch processing")
        sys.exit(1)
    
    if not input_paths:
        print("Error: No input files found")
        sys.exit(1)
    
    print(f"\nProcessing {len(input_paths)} file(s)...")
    
    # Process batch
    results = pipeline.process_batch(
        input_paths,
        model_name=args.model,
        output_dir=args.output_dir,
        prompt=args.prompt
    )
    
    # Print summary
    print("\n" + "=" * 80)
    print("Batch Processing Complete")
    print("=" * 80)
    print(f"Total files: {len(results)}")
    
    success_count = sum(1 for r in results if 'error' not in r)
    error_count = len(results) - success_count
    
    print(f"Successful: {success_count}")
    print(f"Failed: {error_count}")
    
    if args.output_dir:
        print(f"Output directory: {args.output_dir}")
    
    print("=" * 80 + "\n")
    
    # Cleanup
    pipeline.cleanup()


def main():
    """Main entry point for the PDF OCR application."""
    parser = argparse.ArgumentParser(
        description="PDF OCR using Large Language Models",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    
    parser.add_argument(
        "--show-device-info",
        action="store_true",
        help="Show device information (GPUs)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # List models command
    list_parser = subparsers.add_parser("list-models", help="List all available models")
    
    # Process single file command
    process_parser = subparsers.add_parser("process", help="Process a single PDF or image file")
    process_parser.add_argument("input", type=str, help="Path to input PDF or image file")
    process_parser.add_argument("--model", type=str, required=True, help="Model name to use")
    process_parser.add_argument("--output", type=str, help="Path to output text file")
    process_parser.add_argument("--output-dir", type=str, help="Directory to save output")
    process_parser.add_argument("--save-images", action="store_true", help="Save intermediate images (PDF only)")
    process_parser.add_argument("--images-dir", type=str, help="Directory to save images")
    process_parser.add_argument("--prompt", type=str, help="Custom prompt for the model")
    
    # Batch processing command
    batch_parser = subparsers.add_parser("batch", help="Process multiple files in batch")
    batch_parser.add_argument("--model", type=str, required=True, help="Model name to use")
    batch_parser.add_argument("--input-dir", type=str, help="Directory containing input files")
    batch_parser.add_argument("--input-files", nargs="+", help="List of input files")
    batch_parser.add_argument("--output-dir", type=str, required=True, help="Directory to save outputs")
    batch_parser.add_argument("--prompt", type=str, help="Custom prompt for the model")
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Execute command
    if args.command == "list-models":
        list_models(args)
    elif args.command == "process":
        process_file(args)
    elif args.command == "batch":
        process_batch(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

