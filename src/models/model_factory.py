from typing import Dict, Any
import logging

from .base_model import BaseOCRModel
from .qwen25vl_model import Qwen25VLModel

logger = logging.getLogger(__name__)


class ModelFactory:
    """Factory class for creating OCR model instances."""
    
    MODEL_TYPES = {
        'qwen25vl': Qwen25VLModel
    }
    
    @staticmethod
    def create_model(
        model_config: Dict[str, Any],
        device_manager,
        inference_config: Dict[str, Any]
    ) -> BaseOCRModel:
        """
        Create an OCR model instance based on configuration.
        
        Args:
            model_config: Model configuration dictionary
            device_manager: Device manager instance
            inference_config: Inference configuration dictionary
            
        Returns:
            Instance of the appropriate OCR model
        """
        model_type = model_config.get('type')
        
        if model_type not in ModelFactory.MODEL_TYPES:
            raise ValueError(f"Unknown model type: {model_type}. Available types: {list(ModelFactory.MODEL_TYPES.keys())}")
        
        model_class = ModelFactory.MODEL_TYPES[model_type]
        logger.info(f"Creating model instance for type: {model_type}")
        
        return model_class(model_config, device_manager, inference_config)
    
    @staticmethod
    def get_available_model_types() -> list:
        """
        Get list of available model types.
        
        Returns:
            List of model type strings
        """
        return list(ModelFactory.MODEL_TYPES.keys())

