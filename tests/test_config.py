"""
Unit tests for configuration management.
"""

import unittest
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config.config_manager import ConfigManager


class TestConfigManager(unittest.TestCase):
    """Test cases for ConfigManager class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config_manager = ConfigManager("config.yaml")
    
    def test_load_config(self):
        """Test configuration file loading."""
        self.assertIsNotNone(self.config_manager.config)
        self.assertIn('models', self.config_manager.config)
        self.assertIn('device', self.config_manager.config)
        self.assertIn('inference', self.config_manager.config)
    
    def test_get_all_models(self):
        """Test retrieving all models."""
        models = self.config_manager.get_all_models()
        self.assertIsInstance(models, list)
        self.assertGreater(len(models), 0)
        
        for model in models:
            self.assertIn('name', model)
            self.assertIn('model_id', model)
            self.assertIn('type', model)
    
    def test_get_model_by_name(self):
        """Test retrieving a specific model by name."""
        model = self.config_manager.get_model_by_name("Qwen2.5-VL-7B")
        self.assertIsNotNone(model)
        self.assertEqual(model['name'], "Qwen2.5-VL-7B")
        self.assertEqual(model['type'], "qwen2vl")
    
    def test_get_nonexistent_model(self):
        """Test retrieving a non-existent model."""
        model = self.config_manager.get_model_by_name("NonExistentModel")
        self.assertIsNone(model)
    
    def test_get_models_by_type(self):
        """Test retrieving models by type."""
        qwen_models = self.config_manager.get_models_by_type("qwen2vl")
        self.assertIsInstance(qwen_models, list)
        self.assertGreater(len(qwen_models), 0)
        
        for model in qwen_models:
            self.assertEqual(model['type'], "qwen2vl")
    
    def test_get_inference_config(self):
        """Test retrieving inference configuration."""
        inference_config = self.config_manager.get_inference_config()
        self.assertIsInstance(inference_config, dict)
        self.assertIn('batch_size', inference_config)
        self.assertIn('max_new_tokens', inference_config)
    
    def test_get_device_config(self):
        """Test retrieving device configuration."""
        device_config = self.config_manager.get_device_config()
        self.assertIsInstance(device_config, dict)
        self.assertIn('use_multi_gpu', device_config)
        self.assertIn('device_map', device_config)
    
    def test_get_ocr_config(self):
        """Test retrieving OCR configuration."""
        ocr_config = self.config_manager.get_ocr_config()
        self.assertIsInstance(ocr_config, dict)
        self.assertIn('dpi', ocr_config)
        self.assertIn('output_format', ocr_config)


if __name__ == '__main__':
    unittest.main()

