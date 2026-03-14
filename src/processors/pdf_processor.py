from pathlib import Path
from typing import List, Dict, Any, Union, Optional
from PIL import Image
import pdf2image
import logging

logger = logging.getLogger(__name__)


class PDFProcessor:
    """Handles PDF to image conversion for OCR processing."""
    
    def __init__(self, ocr_config: Dict[str, Any]):
        """
        Initialize the PDF processor.
        
        Args:
            ocr_config: OCR configuration dictionary
        """
        self.dpi = ocr_config.get('dpi', 300)
        self.image_quality = ocr_config.get('image_quality', 95)
        self.max_image_size = ocr_config.get('max_image_size', 1536)

    def resize_for_model(self, image: Image.Image, max_side: Optional[int] = None) -> Image.Image:
        """
        Resize image so longest side <= max_side; preserves aspect ratio.
        If max_side is None, use config max_image_size. If max_side <= 0, return image unchanged.
        """
        effective = max_side if max_side is not None else self.max_image_size
        if effective <= 0:
            return image
        w, h = image.size
        if max(w, h) <= effective:
            return image
        if w >= h:
            new_w = effective
            new_h = int(h * (effective / w))
        else:
            new_h = effective
            new_w = int(w * (effective / h))
        return image.resize((new_w, new_h), Image.Resampling.LANCZOS)

    def pdf_to_images(
        self, pdf_path: Union[str, Path], max_image_size_override: Optional[int] = None
    ) -> List[Image.Image]:
        """
        Convert a PDF file to a list of PIL Images.
        
        Args:
            pdf_path: Path to the PDF file
            max_image_size_override: If set, use for resize: 0 = no resize, >0 = max longest side.
                If None, use config max_image_size (resize when > 0).
            
        Returns:
            List of PIL Image objects, one per page
        """
        pdf_path = Path(pdf_path)
        
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        if not pdf_path.suffix.lower() == '.pdf':
            raise ValueError(f"File is not a PDF: {pdf_path}")
        
        logger.info(f"Converting PDF to images: {pdf_path}")
        
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(
                pdf_path,
                dpi=self.dpi,
                fmt='PNG'
            )
            # Optionally resize high-resolution pages (keeps aspect ratio)
            max_side = max_image_size_override if max_image_size_override is not None else self.max_image_size
            if max_side > 0:
                images = [self.resize_for_model(im, max_side) for im in images]
                logger.info(f"Resized pages to max longest side {max_side}px")
            logger.info(f"Successfully converted {len(images)} page(s) from PDF")
            return images
            
        except Exception as e:
            logger.error(f"Failed to convert PDF to images: {e}")
            raise
    
    def save_images(self, images: List[Image.Image], output_dir: Union[str, Path], prefix: str = "page") -> List[Path]:
        """
        Save images to disk.
        
        Args:
            images: List of PIL Image objects
            output_dir: Directory to save images
            prefix: Prefix for image filenames
            
        Returns:
            List of paths to saved images
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        saved_paths = []
        
        for idx, image in enumerate(images, start=1):
            output_path = output_dir / f"{prefix}_{idx:04d}.png"
            image.save(output_path, quality=self.image_quality)
            saved_paths.append(output_path)
            logger.debug(f"Saved image: {output_path}")
        
        logger.info(f"Saved {len(saved_paths)} image(s) to {output_dir}")
        return saved_paths
    
    def load_image(self, image_path: Union[str, Path]) -> Image.Image:
        """
        Load a single image from disk.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            PIL Image object
        """
        image_path = Path(image_path)
        
        if not image_path.exists():
            raise FileNotFoundError(f"Image file not found: {image_path}")
        
        try:
            image = Image.open(image_path)
            # Convert to RGB if necessary
            if image.mode != 'RGB':
                image = image.convert('RGB')
            return image
        except Exception as e:
            logger.error(f"Failed to load image: {e}")
            raise
    
    def load_images_from_directory(self, directory: Union[str, Path], extensions: List[str] = None) -> List[Image.Image]:
        """
        Load all images from a directory.
        
        Args:
            directory: Path to the directory
            extensions: List of file extensions to include (default: ['.png', '.jpg', '.jpeg', '.tiff', '.bmp'])
            
        Returns:
            List of PIL Image objects
        """
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.tiff', '.bmp']
        
        directory = Path(directory)
        
        if not directory.exists():
            raise FileNotFoundError(f"Directory not found: {directory}")
        
        if not directory.is_dir():
            raise ValueError(f"Path is not a directory: {directory}")
        
        images = []
        image_files = []
        
        # Collect all image files
        for ext in extensions:
            image_files.extend(directory.glob(f"*{ext}"))
            image_files.extend(directory.glob(f"*{ext.upper()}"))
        
        # Sort files by name
        image_files = sorted(set(image_files))
        
        # Load images
        for image_file in image_files:
            try:
                image = self.load_image(image_file)
                images.append(image)
            except Exception as e:
                logger.warning(f"Failed to load {image_file}: {e}")
        
        logger.info(f"Loaded {len(images)} image(s) from {directory}")
        return images

