import torch
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class DeviceManager:
    """Manages device allocation for model inference with GPU support."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize device manager.
        
        Args:
            config: Device configuration dictionary
        """
        self.use_multi_gpu = config.get('use_multi_gpu', True)
        self.device_map = config.get('device_map', 'auto')
        self.gpu_memory_utilization = config.get('gpu_memory_utilization', 0.9)
        
        self.available_devices = self._detect_devices()
        
    def _detect_devices(self) -> Dict[str, Any]:
        """
        Detect available devices (GPUs).
        
        Returns:
            Dictionary containing device information
        """
        devices = {
            'cuda_available': torch.cuda.is_available(),
            'cuda_device_count': 0,
            'cpu_available': True
        }
        
        if torch.cuda.is_available():
            devices['cuda_device_count'] = torch.cuda.device_count()
            logger.info(f"Detected {devices['cuda_device_count']} CUDA device(s)")
            for i in range(devices['cuda_device_count']):
                device_name = torch.cuda.get_device_name(i)
                logger.info(f"  GPU {i}: {device_name}")
        
        return devices
    
    def get_device_map(self) -> str:
        """
        Get the appropriate device map for model loading.
        
        Returns:
            Device map string ('auto', 'cuda', or 'cpu')
        """
        if self.use_multi_gpu and self.available_devices['cuda_device_count'] > 1:
            return 'auto'
        elif self.available_devices['cuda_available']:
            return 'cuda:0'
        else:
            return 'cpu'
    
    def get_device(self) -> torch.device:
        """
        Get the primary device for operations.
        
        Returns:
            PyTorch device object
        """
        if self.available_devices['cuda_available']:
            return torch.device('cuda:0')
        
        return torch.device('cpu')
    
    def get_model_kwargs(self) -> Dict[str, Any]:
        """
        Get keyword arguments for model loading with device configuration.
        
        Returns:
            Dictionary of model loading kwargs
        """
        kwargs = {}
        
        device_map = self.get_device_map()
        kwargs['device_map'] = device_map
        
        # Add memory optimization for GPUs
        if self.available_devices['cuda_available']:
            kwargs['torch_dtype'] = torch.bfloat16
            kwargs['low_cpu_mem_usage'] = True
        
        return kwargs
    
    def print_device_info(self) -> None:
        """Print detailed device information."""
        print("=" * 60)
        print("Device Information:")
        print("=" * 60)
        print(f"CPU Available: {self.available_devices['cpu_available']}")
        print(f"CUDA Available: {self.available_devices['cuda_available']}")
        if self.available_devices['cuda_available']:
            print(f"CUDA Devices: {self.available_devices['cuda_device_count']}")
            for i in range(self.available_devices['cuda_device_count']):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"Selected Device Map: {self.get_device_map()}")
        print("=" * 60)

