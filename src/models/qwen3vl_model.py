from typing import Any, Dict, List, Optional
from PIL import Image
import torch
from transformers import AutoProcessor
import logging
import os
from dotenv import load_dotenv

from .base_model import BaseOCRModel

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Qwen3VLModel(BaseOCRModel):
    """Qwen 3 VL model handler for OCR tasks."""
    
    @staticmethod
    def _clean_code_fences(text: str) -> str:
        """Remove markdown code fences from output."""
        import re
        # Remove ```markdown ... ``` or ``` ... ``` blocks
        text = re.sub(r'^```(?:markdown)?\s*\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n```\s*$', '', text, flags=re.MULTILINE)
        return text.strip()
    
    def __init__(self, model_config: Dict[str, Any], device_manager, inference_config: Dict[str, Any]):
        """
        Initialize the Qwen 3 VL model.
        
        Args:
            model_config: Model configuration dictionary
            device_manager: Device manager instance
            inference_config: Inference configuration dictionary
        """
        super().__init__(model_config, device_manager)
        self.inference_config = inference_config
        self.max_pixels = model_config.get('max_pixels', 1280)
        self.min_pixels = model_config.get('min_pixels', 256)
        
    def load_model(self) -> None:
        """Load the Qwen 3 VL model and processor."""
        logger.info(f"Loading Qwen 3 VL model: {self.model_config['model_id']}")
        
        try:
            # Get HuggingFace token from environment if available
            hf_token = os.getenv('HF_TOKEN')
            if hf_token:
                logger.info("Using HuggingFace token from environment")
            
            # Load processor first
            self.processor = AutoProcessor.from_pretrained(
                self.model_config['model_id'],
                trust_remote_code=True,
                token=hf_token
            )
            
            # Get device configuration
            model_kwargs = self.device_manager.get_model_kwargs()
            # Use Flash Attention 2 when available (faster + better precision)
            try:
                import flash_attn
                model_kwargs['attn_implementation'] = 'flash_attention_2'
                logger.info("Flash Attention 2 is available, enabling it for Qwen3-VL")
            except ImportError:
                logger.warning("Flash Attention 2 not installed; using default attention (slower, consider installing flash-attn)")
            
            # Import and load Qwen3VL model
            try:
                from transformers import Qwen3VLForConditionalGeneration
                logger.info("Using Qwen3VLForConditionalGeneration")
                
                # Use dtype="auto" as recommended in docs
                if 'torch_dtype' in model_kwargs:
                    model_kwargs['dtype'] = model_kwargs.pop('torch_dtype')
                else:
                    model_kwargs['dtype'] = "auto"
                
                self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                    self.model_config['model_id'],
                    trust_remote_code=True,
                    token=hf_token,
                    **model_kwargs
                )
            except ImportError:
                # Fallback to AutoModel if Qwen3VL not available
                logger.warning("Qwen3VLForConditionalGeneration not found, using AutoModel")
                from transformers import AutoModelForImageTextToText
                
                self.model = AutoModelForImageTextToText.from_pretrained(
                    self.model_config['model_id'],
                    trust_remote_code=True,
                    token=hf_token,
                    **model_kwargs
                )
            
            # Set model to evaluation mode
            self.model.eval()

            # Optional: torch.compile to reduce Python overhead (PyTorch 2.0+, first run compiles)
            if self.inference_config.get('use_torch_compile', False):
                compile_fn = getattr(torch, 'compile', None)
                if compile_fn is not None:
                    try:
                        self.model = compile_fn(self.model, mode='reduce-overhead')
                        logger.info("torch.compile enabled (mode=reduce-overhead)")
                    except Exception as e:
                        logger.warning(f"torch.compile failed, using plain model: {e}")
                else:
                    logger.warning("use_torch_compile is true but torch.compile not available (PyTorch 2.0+)")

            # Log actual model device (critical: must be GPU for speed)
            try:
                model_dev = next(self.model.parameters()).device
                if str(model_dev).startswith('cuda'):
                    logger.info(f"Model is on GPU: {model_dev}")
                else:
                    logger.warning(f"Model is on CPU: {model_dev} — inference will be very slow. Check device_map and CUDA.")
            except Exception as e:
                logger.warning(f"Could not determine model device: {e}")
            
            logger.info(f"Successfully loaded {self.model_config['name']}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def warmup(self) -> None:
        """Run a minimal inference to initialize CUDA and avoid slow first real inference."""
        if self.model is None or self.processor is None:
            return
        try:
            warmup_image = Image.new("RGB", (64, 64), color=(255, 255, 255))
            messages = [
                {"role": "user", "content": [{"type": "image", "image": warmup_image}, {"type": "text", "text": "Hi"}]}
            ]
            inputs = self.processor.apply_chat_template(
                messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )
            model_dev = next(self.model.parameters()).device
            if hasattr(inputs, 'to'):
                inputs = inputs.to(model_dev)
            else:
                inputs = {k: v.to(model_dev) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=5)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("Qwen3-VL warmup done.")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")

    def process_image(self, image: Image.Image, prompt: str = None) -> str:
        """
        Process a single image and extract text using Qwen 3 VL.
        
        Args:
            image: PIL Image object
            prompt: Optional prompt for the model
            
        Returns:
            Extracted text from the image
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Default OCR prompt if none provided
        if prompt is None:
            prompt = "Extract all text from this image. Preserve the layout, formatting, tables, and structure. Use Markdown syntax for headings, lists, tables, bold, italic, etc. Output only the formatted text without code blocks or fences."
        
        # Prepare messages for the model (Qwen 3 VL format)
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Preparation for inference (following official docs)
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt"
        )
        
        # Move inputs to device (must match model device)
        model_dev = next(self.model.parameters()).device
        if hasattr(inputs, 'to'):
            inputs = inputs.to(model_dev)
        else:
            inputs = {k: v.to(model_dev) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        if str(model_dev).startswith('cuda'):
            logger.info(f"Qwen3-VL process_image: running on GPU {model_dev} (image size {image.size})")
        else:
            logger.warning(f"Qwen3-VL process_image: model on CPU {model_dev} — inference will be very slow")
        
        # Generate (following official docs)
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.inference_config.get('max_new_tokens', 2048)
            )
        
        # Trim generated tokens to remove input (following official docs)
        input_ids_tensor = inputs.input_ids if hasattr(inputs, 'input_ids') else inputs['input_ids']
        generated_ids_trimmed = [
            out_ids[len(in_seq):] for in_seq, out_ids in zip(input_ids_tensor, generated_ids)
        ]
        
        # Decode output (following official docs)
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
        # Clean up GPU memory
        del inputs
        del generated_ids
        del generated_ids_trimmed
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clean up markdown code fences if present
        output_text = self._clean_code_fences(output_text)
        
        return output_text
    
    def process_batch(self, images: List[Image.Image], prompts: List[str] = None) -> List[str]:
        """
        Process a batch of images.
        
        Args:
            images: List of PIL Image objects
            prompts: Optional list of prompts for each image
            
        Returns:
            List of extracted texts
        """
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Default prompts if none provided
        if prompts is None:
            prompts = ["Extract all text from this image. Preserve the layout, formatting, tables, and structure. Use Markdown syntax for headings, lists, tables, bold, italic, etc. Output only the formatted text without code blocks or fences."] * len(images)
        
        if len(prompts) != len(images):
            raise ValueError("Number of prompts must match number of images")
        
        results = []
        batch_size = self.inference_config.get('batch_size', 1)
        
        # Process in batches
        for i in range(0, len(images), batch_size):
            batch_images = images[i:i + batch_size]
            batch_prompts = prompts[i:i + batch_size]
            
            # Prepare messages for each image in batch
            all_messages = []
            for img, prompt in zip(batch_images, batch_prompts):
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "image": img},
                            {"type": "text", "text": prompt}
                        ]
                    }
                ]
                all_messages.append(messages)
            
            model_dev = next(self.model.parameters()).device
            if str(model_dev).startswith('cuda'):
                logger.info(f"Qwen3-VL process_batch: batch {i // batch_size + 1}, size {len(batch_images)}, device={model_dev}")
            else:
                logger.warning(f"Qwen3-VL process_batch: model on CPU {model_dev} — inference will be very slow")
            
            # Process each image
            for messages in all_messages:
                try:
                    # Prepare inputs
                    inputs = self.processor.apply_chat_template(
                        messages,
                        tokenize=True,
                        add_generation_prompt=True,
                        return_dict=True,
                        return_tensors="pt"
                    )
                    
                    # Move to device
                    inputs = inputs.to(model_dev)
                    
                    # Generate
                    with torch.no_grad():
                        generated_ids = self.model.generate(
                            **inputs,
                            max_new_tokens=self.inference_config.get('max_new_tokens', 2048)
                        )
                    
                    # Trim generated tokens
                    input_ids_tensor = inputs.input_ids if hasattr(inputs, 'input_ids') else inputs['input_ids']
                    generated_ids_trimmed = [
                        out_ids[len(in_seq):] for in_seq, out_ids in zip(input_ids_tensor, generated_ids)
                    ]
                    
                    # Decode output
                    output_text = self.processor.batch_decode(
                        generated_ids_trimmed,
                        skip_special_tokens=True,
                        clean_up_tokenization_spaces=False
                    )[0]
                    
                    # Clean up GPU memory
                    del inputs
                    del generated_ids
                    del generated_ids_trimmed
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    
                    # Clean up code fences
                    output_text = self._clean_code_fences(output_text)
                    
                    results.append(output_text)
                    
                except Exception as e:
                    logger.error(f"Error processing image in batch: {e}")
                    results.append(f"Error: {str(e)}")
        
        return results

