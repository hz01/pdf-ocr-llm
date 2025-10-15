import argparse
import logging
from pathlib import Path
import sys
import multiprocessing
import time

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


def launch_ui(args) -> None:
    """Launch the Gradio web UI."""
    from src.ui.app import demo
    
    print("\n" + "=" * 80)
    print("Starting Gradio Web UI")
    print("=" * 80)
    print(f"Server: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop")
    print("=" * 80 + "\n")
    
    demo.launch(
        server_name=args.host,
        server_port=args.port,
        share=args.share
    )


def launch_api(args) -> None:
    """Launch the FastAPI REST API server."""
    import uvicorn
    from src.api.app import app
    
    print("\n" + "=" * 80)
    print("Starting FastAPI REST API")
    print("=" * 80)
    print(f"Server: http://{args.host}:{args.port}")
    print(f"API Docs: http://{args.host}:{args.port}/docs")
    print("Press Ctrl+C to stop")
    print("=" * 80 + "\n")
    
    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level=args.log_level.lower()
    )


def _run_api_process(host: str, port: int, log_level: str) -> None:
    """Helper function to run API in a separate process."""
    import uvicorn
    from src.api.app import app
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level=log_level.lower()
    )


def _run_ui_process(host: str, port: int) -> None:
    """Helper function to run UI in a separate process."""
    from src.ui.app import demo
    
    demo.launch(
        server_name=host,
        server_port=port,
        share=False
    )


def launch_both(args) -> None:
    """Launch both API and UI servers together."""
    print("\n" + "=" * 80)
    print("Starting PDF OCR Services")
    print("=" * 80)
    print(f"Web UI:  http://{args.ui_host}:{args.ui_port}")
    print(f"API:     http://{args.api_host}:{args.api_port}")
    print(f"API Docs: http://{args.api_host}:{args.api_port}/docs")
    print("\nPress Ctrl+C to stop all services")
    print("=" * 80 + "\n")
    
    # Create processes for API and UI
    api_process = multiprocessing.Process(
        target=_run_api_process,
        args=(args.api_host, args.api_port, args.log_level)
    )
    ui_process = multiprocessing.Process(
        target=_run_ui_process,
        args=(args.ui_host, args.ui_port)
    )
    
    try:
        # Start both processes
        api_process.start()
        time.sleep(2)  # Give API a moment to start
        ui_process.start()
        
        # Wait for both processes
        api_process.join()
        ui_process.join()
        
    except KeyboardInterrupt:
        print("\n\nShutting down services...")
        api_process.terminate()
        ui_process.terminate()
        api_process.join()
        ui_process.join()
        print("All services stopped.")
    except Exception as e:
        print(f"\nError: {e}")
        api_process.terminate()
        ui_process.terminate()


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
    
    # Web UI command
    ui_parser = subparsers.add_parser("ui", help="Launch Gradio web interface")
    ui_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    ui_parser.add_argument("--port", type=int, default=7860, help="Port number (default: 7860)")
    ui_parser.add_argument("--share", action="store_true", help="Create public share link")
    
    # API command
    api_parser = subparsers.add_parser("api", help="Launch FastAPI REST API server")
    api_parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address (default: 0.0.0.0)")
    api_parser.add_argument("--port", type=int, default=8000, help="Port number (default: 8000)")
    
    # Serve both API and UI
    serve_parser = subparsers.add_parser("serve", help="Launch both API and UI servers")
    serve_parser.add_argument("--api-host", type=str, default="0.0.0.0", help="API host address (default: 0.0.0.0)")
    serve_parser.add_argument("--api-port", type=int, default=8000, help="API port number (default: 8000)")
    serve_parser.add_argument("--ui-host", type=str, default="0.0.0.0", help="UI host address (default: 0.0.0.0)")
    serve_parser.add_argument("--ui-port", type=int, default=7860, help="UI port number (default: 7860)")
    
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
    elif args.command == "ui":
        launch_ui(args)
    elif args.command == "api":
        launch_api(args)
    elif args.command == "serve":
        launch_both(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

