from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union
from PIL import Image
import torch


class BaseOCRModel(ABC):
    """Abstract base class for OCR models."""
    
    def __init__(self, model_config: Dict[str, Any], device_manager):
        """
        Initialize the base OCR model.
        
        Args:
            model_config: Model configuration dictionary
            device_manager: Device manager instance
        """
        self.model_config = model_config
        self.device_manager = device_manager
        self.model = None
        self.processor = None
        self.tokenizer = None
        
    @abstractmethod
    def load_model(self) -> None:
        """Load the model and associated components."""
        pass
    
    @abstractmethod
    def process_image(self, image: Image.Image, prompt: str = None) -> str:
        """
        Process a single image and extract text.
        
        Args:
            image: PIL Image object
            prompt: Optional prompt for the model
            
        Returns:
            Extracted text from the image
        """
        pass
    
    @abstractmethod
    def process_batch(self, images: List[Image.Image], prompts: List[str] = None) -> List[str]:
        """
        Process a batch of images.
        
        Args:
            images: List of PIL Image objects
            prompts: Optional list of prompts for each image
            
        Returns:
            List of extracted texts
        """
        pass
    
    def unload_model(self) -> None:
        """Unload the model from memory."""
        if self.model is not None:
            del self.model
            self.model = None
        if self.processor is not None:
            del self.processor
            self.processor = None
        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None
        
        # Clear GPU cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the loaded model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            'name': self.model_config.get('name'),
            'model_id': self.model_config.get('model_id'),
            'type': self.model_config.get('type'),
            'loaded': self.model is not None
        }

