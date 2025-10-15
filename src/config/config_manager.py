import yaml
from pathlib import Path
from typing import Dict, List, Any, Optional


class ConfigManager:
    """Manages configuration loading and validation."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        Initialize the configuration manager.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Dictionary containing configuration
        """
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def get_all_models(self) -> List[Dict[str, Any]]:
        """
        Get all available models from configuration.
        
        Returns:
            List of model configurations
        """
        all_models = []
        for model_type, models in self.config.get('models', {}).items():
            all_models.extend(models)
        return all_models
    
    def get_model_by_name(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get model configuration by name.
        
        Args:
            model_name: Name of the model
            
        Returns:
            Model configuration dictionary or None if not found
        """
        for model in self.get_all_models():
            if model['name'] == model_name:
                return model
        return None
    
    def get_models_by_type(self, model_type: str) -> List[Dict[str, Any]]:
        """
        Get all models of a specific type.
        
        Args:
            model_type: Type of model (e.g., 'qwen2vl')
            
        Returns:
            List of model configurations
        """
        return [model for model in self.get_all_models() if model['type'] == model_type]
    
    def get_inference_config(self) -> Dict[str, Any]:
        """
        Get inference configuration.
        
        Returns:
            Inference configuration dictionary
        """
        return self.config.get('inference', {})
    
    def get_device_config(self) -> Dict[str, Any]:
        """
        Get device configuration.
        
        Returns:
            Device configuration dictionary
        """
        return self.config.get('device', {})
    
    def get_ocr_config(self) -> Dict[str, Any]:
        """
        Get OCR configuration.
        
        Returns:
            OCR configuration dictionary
        """
        return self.config.get('ocr', {})

