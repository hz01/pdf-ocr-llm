from typing import Any, Dict, List, Optional
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import logging
import os
from dotenv import load_dotenv

from .base_model import BaseOCRModel

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)

# InternVL image preprocessing constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


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
    
    @staticmethod
    def build_transform(input_size):
        """Build image transformation pipeline (from InternVL docs)."""
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform
    
    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
        """Find the closest aspect ratio for dynamic preprocessing (from InternVL docs)."""
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    @staticmethod
    def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
        """Dynamic image preprocessing with tiling (from InternVL docs)."""
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = InternVLModel.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images
    
    def load_image(self, image: Image.Image, input_size=448, max_num=12):
        """Load and preprocess image for InternVL (from official docs)."""
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(img) for img in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values
    
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
            
            # Check for flash attention availability
            try:
                import flash_attn
                logger.info("Flash Attention 2 is available, enabling it")
                use_flash_attn = True
            except ImportError:
                logger.warning("Flash Attention 2 not installed, using standard attention (slower)")
                use_flash_attn = False
            
            # Load model - InternVL uses AutoModel with use_flash_attn parameter
            self.model = AutoModel.from_pretrained(
                self.model_config['model_id'],
                torch_dtype=torch.bfloat16,
                low_cpu_mem_usage=True,
                use_flash_attn=use_flash_attn,
                trust_remote_code=True,
                token=hf_token,
                device_map=model_kwargs.get('device_map', 'auto')
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
        
        # Preprocess image with dynamic tiling (following official InternVL docs)
        logger.info(f"Preprocessing image with InternVL dynamic tiling...")
        pixel_values = self.load_image(image, input_size=448, max_num=12)
        pixel_values = pixel_values.to(torch.bfloat16).to(self.model.device)
        logger.info(f"Created {pixel_values.shape[0]} image tiles")
        
        # Prepare generation config (following official docs)
        generation_config = dict(
            max_new_tokens=self.inference_config.get('max_new_tokens', 2048),
            do_sample=True
        )
        
        # Format question with image placeholder (following official docs)
        question = f'<image>\n{prompt}'
        
        try:
            # Call InternVL chat method (following official docs)
            logger.info(f"Calling InternVL chat...")
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config
            )
            
            # Handle response - InternVL returns string directly or tuple with history
            if isinstance(response, tuple):
                output_text = response[0]
                logger.info(f"Got tuple response with history: {len(output_text)} chars")
            elif isinstance(response, str):
                output_text = response
                logger.info(f"Got string response: {len(output_text)} chars")
            else:
                logger.warning(f"Unexpected response type: {type(response)}")
                output_text = str(response)
            
            # Check if we got a valid response
            if not output_text or len(output_text.strip()) == 0:
                logger.error("InternVL returned empty response!")
                raise ValueError("Empty response from model")
                
            logger.info(f"Successfully extracted {len(output_text)} characters")
            
        except Exception as e:
            logger.error(f"Error during InternVL chat: {e}")
            raise RuntimeError(f"InternVL failed to process image: {e}")
        finally:
            # Clean up GPU memory
            del pixel_values
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



