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
        
        # Use InternVL's chat interface
        generation_config = dict(
            max_new_tokens=self.inference_config.get('max_new_tokens', 2048),
            do_sample=True if self.inference_config.get('temperature', 0.1) > 0 else False,
            temperature=self.inference_config.get('temperature', 0.1),
            top_p=self.inference_config.get('top_p', 0.9),
        )
        
        # InternVL models typically have a chat method
        try:
            output_text = self.model.chat(
                self.tokenizer,
                pixel_values=None,
                question=prompt,
                generation_config=generation_config,
                history=None,
                return_history=False,
                IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
                IMG_START_TOKEN='<img>',
                IMG_END_TOKEN='</img>',
                verbose=False,
                image=image
            )
        except AttributeError:
            # Fallback to standard generation if chat method not available
            logger.warning("Model doesn't have chat method, using standard generation")
            pixel_values = self.model.extract_feature(image)
            question = f"<image>\n{prompt}"
            
            input_ids = self.tokenizer(question, return_tensors='pt').input_ids
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            
            with torch.no_grad():
                generated_ids = self.model.generate(
                    input_ids=input_ids,
                    pixel_values=pixel_values.to(device) if pixel_values is not None else None,
                    **generation_config
                )
            
            output_text = self.tokenizer.decode(
                generated_ids[0],
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )
            
            # Remove the prompt from output if present
            if prompt in output_text:
                output_text = output_text.replace(prompt, "").strip()
        
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



