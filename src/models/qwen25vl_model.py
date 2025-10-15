from typing import Any, Dict, List, Optional
from PIL import Image
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor
import logging
import os
from dotenv import load_dotenv

from .base_model import BaseOCRModel

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class Qwen25VLModel(BaseOCRModel):
    """Qwen 2.5 VL model handler for OCR tasks."""
    
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
        Initialize the Qwen 2.5 VL model.
        
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
        """Load the Qwen 2.5 VL model and processor."""
        logger.info(f"Loading Qwen 2.5 VL model: {self.model_config['model_id']}")
        
        try:
            # Get HuggingFace token from environment if available
            hf_token = os.getenv('HF_TOKEN')
            if hf_token:
                logger.info("Using HuggingFace token from environment")
            
            # Load processor
            self.processor = AutoProcessor.from_pretrained(
                self.model_config['model_id'],
                trust_remote_code=True,
                max_pixels=self.max_pixels * 28 * 28,
                min_pixels=self.min_pixels * 28 * 28,
                token=hf_token,
                use_fast=False  # Avoid fast tokenizer issues
            )
            
            # Get device configuration
            model_kwargs = self.device_manager.get_model_kwargs()
            
            # Load model
            self.model = AutoModelForImageTextToText.from_pretrained(
                self.model_config['model_id'],
                trust_remote_code=True,
                token=hf_token,
                **model_kwargs
            )
            
            # Set model to evaluation mode
            self.model.eval()
            
            logger.info(f"Successfully loaded {self.model_config['name']}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def process_image(self, image: Image.Image, prompt: str = None) -> str:
        """
        Process a single image and extract text using Qwen 2.5 VL.
        
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
        
        # Prepare messages for the model
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
        
        # Process input
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        inputs = self.processor(
            text=[text],
            images=[image],
            padding=True,
            return_tensors="pt"
        )
        
        # Move inputs to device
        device = self.device_manager.get_device()
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=self.inference_config.get('max_new_tokens', 2048),
                temperature=self.inference_config.get('temperature', 0.1),
                top_p=self.inference_config.get('top_p', 0.9),
                do_sample=False
            )
        
        # Trim generated tokens to remove input
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
        ]
        
        # Decode output
        output_text = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]
        
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
            
            # Process each message
            texts = []
            for messages, img in zip(all_messages, batch_images):
                text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                texts.append(text)
            
            # Prepare inputs
            inputs = self.processor(
                text=texts,
                images=batch_images,
                padding=True,
                return_tensors="pt"
            )
            
            # Move inputs to device
            device = self.device_manager.get_device()
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.inference_config.get('max_new_tokens', 2048),
                    temperature=self.inference_config.get('temperature', 0.1),
                    top_p=self.inference_config.get('top_p', 0.9),
                    do_sample=False
                )
            
            # Trim generated tokens
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs['input_ids'], generated_ids)
            ]
            
            # Decode outputs
            batch_outputs = self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Clean up code fences
            batch_outputs = [self._clean_code_fences(text) for text in batch_outputs]
            
            results.extend(batch_outputs)
        
        return results


