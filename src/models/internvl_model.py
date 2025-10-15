from typing import Any, Dict, List, Optional
from PIL import Image
import torch
from transformers import AutoModel, AutoTokenizer
import logging
import os
from dotenv import load_dotenv

from .base_model import BaseOCRModel

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class InternVLModel(BaseOCRModel):
    """InternVL 3.5 model handler for OCR tasks."""
    
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
        Initialize the InternVL 3.5 model.
        
        Args:
            model_config: Model configuration dictionary
            device_manager: Device manager instance
            inference_config: Inference configuration dictionary
        """
        super().__init__(model_config, device_manager)
        self.inference_config = inference_config
        
    def load_model(self) -> None:
        """Load the InternVL 3.5 model and tokenizer."""
        logger.info(f"Loading InternVL 3.5 model: {self.model_config['model_id']}")
        
        try:
            # Get HuggingFace token from environment if available
            hf_token = os.getenv('HF_TOKEN')
            if hf_token:
                logger.info("Using HuggingFace token from environment")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_config['model_id'],
                trust_remote_code=True,
                token=hf_token
            )
            
            # Get device configuration
            model_kwargs = self.device_manager.get_model_kwargs()
            
            # Set default dtype if not already specified
            if 'torch_dtype' not in model_kwargs:
                model_kwargs['torch_dtype'] = torch.bfloat16
            
            # Load model - InternVL uses AutoModel, not AutoModelForVision2Seq
            self.model = AutoModel.from_pretrained(
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
        Process a single image and extract text using InternVL 3.5.
        
        Args:
            image: PIL Image object
            prompt: Optional prompt for the model
            
        Returns:
            Extracted text from the image
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        
        # Default OCR prompt if none provided
        if prompt is None:
            prompt = "Extract all text from this image. Preserve the layout, formatting, tables, and structure. Use Markdown syntax for headings, lists, tables, bold, italic, etc. Output only the formatted text without code blocks or fences."
        
        # Use InternVL's chat interface with correct parameters
        generation_config = dict(
            max_new_tokens=self.inference_config.get('max_new_tokens', 2048),
            do_sample=True if self.inference_config.get('temperature', 0.1) > 0 else False,
        )
        
        if generation_config['do_sample']:
            generation_config['temperature'] = self.inference_config.get('temperature', 0.1)
            generation_config['top_p'] = self.inference_config.get('top_p', 0.9)
        
        # InternVL models have a chat method
        try:
            # Correct chat signature for InternVL
            logger.info(f"Calling InternVL chat with prompt length: {len(prompt)}")
            response = self.model.chat(
                tokenizer=self.tokenizer,
                pixel_values=None,
                question=prompt,
                generation_config=generation_config,
                history=None,
                return_history=False,
                image=image
            )
            
            # Handle both string and tuple responses
            output_text = response[0] if isinstance(response, tuple) else response
            logger.info(f"InternVL response length: {len(output_text) if output_text else 0}")
            
        except Exception as e:
            logger.error(f"Error during InternVL chat: {e}")
            logger.info("Attempting alternative inference method...")
            
            # Alternative approach: prepare messages and use pipeline-like method
            try:
                # Format as conversation
                question = f"<image>\n{prompt}"
                
                # Get model inputs
                inputs = self.tokenizer(question, return_tensors='pt')
                device = next(self.model.parameters()).device
                
                # Move to device
                input_ids = inputs['input_ids'].to(device)
                attention_mask = inputs.get('attention_mask', None)
                if attention_mask is not None:
                    attention_mask = attention_mask.to(device)
                
                # Process image if model has image processing
                if hasattr(self.model, 'encode_images'):
                    image_tensor = self.model.encode_images(image).to(device)
                else:
                    image_tensor = None
                
                # Generate
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        images=image_tensor,
                        **generation_config
                    )
                
                # Decode
                output_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                # Remove prompt from output
                if question in output_text:
                    output_text = output_text.replace(question, "").strip()
                if prompt in output_text:
                    output_text = output_text.replace(prompt, "").strip()
                    
            except Exception as e2:
                logger.error(f"Alternative method also failed: {e2}")
                raise RuntimeError(f"Could not process image with InternVL model: {e}, {e2}")
        
        # Clean up markdown code fences if present
        output_text = self._clean_code_fences(output_text)
        
        return output_text
    
    def process_batch(self, images: List[Image.Image], prompts: List[str] = None) -> List[str]:
        """
        Process a batch of images.
        
        Args:
            images: List of PIL Image objects
            prompts: Optional list of prompts, one per image
            
        Returns:
            List of extracted texts
        """
        if self.model is None or self.tokenizer is None:
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
            
            # Process each image in the batch
            for image, prompt in zip(batch_images, batch_prompts):
                try:
                    result = self.process_image(image, prompt)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing image in batch: {e}")
                    results.append(f"Error: {str(e)}")
        
        return results



