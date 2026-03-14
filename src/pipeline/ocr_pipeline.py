from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from PIL import Image
import logging
from tqdm import tqdm

from ..config.config_manager import ConfigManager
from ..utils.device_manager import DeviceManager
from ..processors.pdf_processor import PDFProcessor
from ..models.model_factory import ModelFactory
from ..models.base_model import BaseOCRModel

logger = logging.getLogger(__name__)


class OCRPipeline:
    """Main OCR pipeline that coordinates PDF processing and model inference."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the OCR pipeline.
        
        Args:
            config_path: Path to the configuration file
        """
        logger.info("Initializing OCR Pipeline")
        
        # Load configuration
        self.config_manager = ConfigManager(config_path)
        
        # Initialize device manager
        device_config = self.config_manager.get_device_config()
        self.device_manager = DeviceManager(device_config)
        
        # Initialize PDF processor
        ocr_config = self.config_manager.get_ocr_config()
        self.pdf_processor = PDFProcessor(ocr_config)
        
        # Get inference configuration
        self.inference_config = self.config_manager.get_inference_config()
        
        # Model will be loaded on demand
        self.current_model: Optional[BaseOCRModel] = None
        self.current_model_name: Optional[str] = None
        
        logger.info("OCR Pipeline initialized successfully")
    
    def list_available_models(self) -> List[Dict[str, Any]]:
        """
        List all available models from configuration.
        
        Returns:
            List of model configuration dictionaries
        """
        return self.config_manager.get_all_models()
    
    def load_model(self, model_name: str, custom_inference_config: Dict[str, Any] = None) -> None:
        """
        Load a specific model by name.
        
        Args:
            model_name: Name of the model to load
            custom_inference_config: Optional custom inference configuration to override defaults
        """
        # Check if model is already loaded
        if self.current_model_name == model_name and self.current_model is not None:
            logger.info(f"Model {model_name} is already loaded")
            return
        
        # Unload current model if any
        if self.current_model is not None:
            logger.info(f"Unloading current model: {self.current_model_name}")
            self.current_model.unload_model()
            self.current_model = None
            self.current_model_name = None
        
        # Get model configuration
        model_config = self.config_manager.get_model_by_name(model_name)
        if model_config is None:
            raise ValueError(f"Model not found: {model_name}")
        
        # Use custom inference config or default
        inference_config = custom_inference_config if custom_inference_config else self.inference_config
        
        # Create and load model
        logger.info(f"Loading model: {model_name}")
        self.current_model = ModelFactory.create_model(
            model_config,
            self.device_manager,
            inference_config
        )
        self.current_model.load_model()
        self.current_model_name = model_name

        # Confirm and log inference device (must be GPU for acceptable speed)
        self.device_manager.log_inference_device()

        # Warmup: run minimal inference so first real page uses GPU immediately
        if hasattr(self.current_model, 'warmup') and callable(self.current_model.warmup):
            logger.info("Running model warmup (one-time CUDA init)...")
            self.current_model.warmup()
            logger.info("Warmup complete.")
        
        logger.info(f"Model {model_name} loaded successfully")
    
    def process_pdf(
        self,
        pdf_path: Union[str, Path],
        model_name: str = None,
        output_path: Union[str, Path] = None,
        save_images: bool = False,
        images_dir: Union[str, Path] = None,
        prompt: str = None,
        temperature: float = None,
        top_p: float = None,
        max_new_tokens: int = None,
        resize_high_res: bool = True
    ) -> Dict[str, Any]:
        """
        Process a PDF file and extract text using OCR.
        
        Args:
            pdf_path: Path to the PDF file
            model_name: Name of the model to use (if not already loaded)
            output_path: Optional path to save the extracted text
            save_images: Whether to save intermediate images
            images_dir: Directory to save images (if save_images is True)
            prompt: Optional prompt for the model
            temperature: Sampling temperature (None = use default)
            top_p: Top-p sampling parameter (None = use default)
            max_new_tokens: Maximum tokens to generate (None = use default)
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        pdf_path = Path(pdf_path)
        
        # Build custom inference config if parameters are provided
        custom_inference_config = None
        if any(param is not None for param in [temperature, top_p, max_new_tokens]):
            custom_inference_config = self.inference_config.copy()
            if temperature is not None:
                custom_inference_config['temperature'] = temperature
            if top_p is not None:
                custom_inference_config['top_p'] = top_p
            if max_new_tokens is not None:
                custom_inference_config['max_new_tokens'] = max_new_tokens
        
        # Load model if specified
        if model_name is not None:
            self.load_model(model_name, custom_inference_config)
        
        # Check if model is loaded
        if self.current_model is None:
            raise RuntimeError("No model loaded. Please specify a model_name or call load_model() first.")
        
        logger.info(f"Processing PDF: {pdf_path} (resize_high_res={resize_high_res})")
        
        # Convert PDF to images; optionally resize high-res pages (0 = no resize)
        max_override = self.pdf_processor.max_image_size if resize_high_res else 0
        images = self.pdf_processor.pdf_to_images(pdf_path, max_image_size_override=max_override)
        
        # Save images if requested
        if save_images:
            if images_dir is None:
                images_dir = pdf_path.parent / f"{pdf_path.stem}_images"
            self.pdf_processor.save_images(images, images_dir)
        
        # Process images with OCR
        num_pages = len(images)
        inference_device = self.device_manager.get_device()
        logger.info(f"Running OCR on {num_pages} page(s), device={inference_device}")
        if inference_device.type != 'cuda':
            logger.warning("OCR is not using CUDA — expect very slow processing. Check GPU and drivers.")
        page_texts = []

        for idx, image in enumerate(tqdm(images, desc="Processing pages"), start=1):
            try:
                logger.info(f"Processing page {idx}/{num_pages}...")
                text = self.current_model.process_image(image, prompt)
                logger.info(f"Page {idx}/{num_pages} done.")
                page_texts.append({
                    'page_number': idx,
                    'text': text
                })
            except Exception as e:
                logger.error(f"Failed to process page {idx}: {e}")
                page_texts.append({
                    'page_number': idx,
                    'text': '',
                    'error': str(e)
                })
        
        # Combine pages: markdown headers for VLMs; raw concatenation for GLM-OCR (its own output)
        model_config = self.config_manager.get_model_by_name(self.current_model_name)
        use_markdown_headers = model_config and model_config.get("type") != "glm_ocr"
        if use_markdown_headers:
            full_text = "\n\n".join([f"# Page {page['page_number']}\n\n{page['text']}" for page in page_texts])
        else:
            full_text = "\n\n".join([page["text"] for page in page_texts])
        
        # Save output if requested
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(full_text)
            logger.info(f"Saved OCR output to: {output_path}")
        
        result = {
            'pdf_path': str(pdf_path),
            'model_name': self.current_model_name,
            'num_pages': len(images),
            'pages': page_texts,
            'full_text': full_text
        }
        
        logger.info(f"Successfully processed PDF: {pdf_path}")
        return result
    
    def process_image(
        self,
        image_path: Union[str, Path],
        model_name: str = None,
        output_path: Union[str, Path] = None,
        prompt: str = None,
        temperature: float = None,
        top_p: float = None,
        max_new_tokens: int = None,
        resize_high_res: bool = True
    ) -> Dict[str, Any]:
        """
        Process a single image and extract text using OCR.
        
        Args:
            image_path: Path to the image file
            model_name: Name of the model to use (if not already loaded)
            output_path: Optional path to save the extracted text
            prompt: Optional prompt for the model
            temperature: Sampling temperature (None = use default)
            top_p: Top-p sampling parameter (None = use default)
            max_new_tokens: Maximum tokens to generate (None = use default)
            resize_high_res: If True, resize images above threshold (faster); if False, use original resolution
            
        Returns:
            Dictionary containing extracted text and metadata
        """
        image_path = Path(image_path)
        
        # Build custom inference config if parameters are provided
        custom_inference_config = None
        if any(param is not None for param in [temperature, top_p, max_new_tokens]):
            custom_inference_config = self.inference_config.copy()
            if temperature is not None:
                custom_inference_config['temperature'] = temperature
            if top_p is not None:
                custom_inference_config['top_p'] = top_p
            if max_new_tokens is not None:
                custom_inference_config['max_new_tokens'] = max_new_tokens
        
        # Load model if specified
        if model_name is not None:
            self.load_model(model_name, custom_inference_config)
        
        # Check if model is loaded
        if self.current_model is None:
            raise RuntimeError("No model loaded. Please specify a model_name or call load_model() first.")
        
        logger.info(f"Processing image: {image_path} (resize_high_res={resize_high_res})")
        
        # Load image
        image = self.pdf_processor.load_image(image_path)
        if resize_high_res:
            image = self.pdf_processor.resize_for_model(image)
        
        # Process image with OCR
        text = self.current_model.process_image(image, prompt)
        
        # Save output if requested
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(text)
            logger.info(f"Saved OCR output to: {output_path}")
        
        result = {
            'image_path': str(image_path),
            'model_name': self.current_model_name,
            'text': text
        }
        
        logger.info(f"Successfully processed image: {image_path}")
        return result
    
    def process_batch(
        self,
        input_paths: List[Union[str, Path]],
        model_name: str = None,
        output_dir: Union[str, Path] = None,
        prompt: str = None
    ) -> List[Dict[str, Any]]:
        """
        Process multiple PDFs or images in batch.
        
        Args:
            input_paths: List of paths to PDF or image files
            model_name: Name of the model to use (if not already loaded)
            output_dir: Optional directory to save extracted texts
            prompt: Optional prompt for the model
            
        Returns:
            List of result dictionaries
        """
        # Load model if specified
        if model_name is not None:
            self.load_model(model_name)
        
        # Check if model is loaded
        if self.current_model is None:
            raise RuntimeError("No model loaded. Please specify a model_name or call load_model() first.")
        
        results = []
        
        for input_path in tqdm(input_paths, desc="Processing files"):
            input_path = Path(input_path)
            
            try:
                if input_path.suffix.lower() == '.pdf':
                    # Determine output path
                    output_path = None
                    if output_dir is not None:
                        output_path = Path(output_dir) / f"{input_path.stem}_ocr.txt"
                    
                    result = self.process_pdf(input_path, output_path=output_path, prompt=prompt)
                else:
                    # Determine output path
                    output_path = None
                    if output_dir is not None:
                        output_path = Path(output_dir) / f"{input_path.stem}_ocr.txt"
                    
                    result = self.process_image(input_path, output_path=output_path, prompt=prompt)
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {input_path}: {e}")
                results.append({
                    'input_path': str(input_path),
                    'error': str(e)
                })
        
        return results
    
    def cleanup(self) -> None:
        """Cleanup resources and unload models."""
        if self.current_model is not None:
            logger.info("Cleaning up resources")
            self.current_model.unload_model()
            self.current_model = None
            self.current_model_name = None

