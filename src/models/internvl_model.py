"""
InternVL 3.5 Instruct model handler (GitHub format).

Uses OpenGVLab/InternVL3_5-*-Instruct with trust_remote_code.
Custom loader: AutoModel + AutoTokenizer, dynamic image preprocessing,
model.chat(tokenizer, pixel_values, question, generation_config).
Includes meta-tensor and tied-weights fixes for device_map="auto".
"""

from typing import Any, Dict, List, Optional
from unittest.mock import patch
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer
import logging
import os
from dotenv import load_dotenv

from .base_model import BaseOCRModel

load_dotenv()
logger = logging.getLogger(__name__)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

DEFAULT_OCR_PROMPT = (
    "Extract all text from this image. Preserve the layout, formatting, tables, and structure. "
    "Use Markdown syntax for headings, lists, tables, bold, italic, etc. "
    "Output only the formatted text without code blocks or fences."
)


class InternVLModel(BaseOCRModel):
    """InternVL 3.5 Instruct (GitHub format) — trust_remote_code, custom preprocessing."""

    @staticmethod
    def _clean_code_fences(text: str) -> str:
        import re
        text = re.sub(r'^```(?:markdown)?\s*\n', '', text, flags=re.MULTILINE)
        text = re.sub(r'\n```\s*$', '', text, flags=re.MULTILINE)
        return text.strip()

    @staticmethod
    def build_transform(input_size: int):
        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        return T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])

    @staticmethod
    def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
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
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1)
            if i * j <= max_num and i * j >= min_num
        )
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        target_aspect_ratio = InternVLModel.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size
        )
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(target_aspect_ratio[0] * target_aspect_ratio[1]):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            processed_images.append(resized_img.crop(box))
        if use_thumbnail and len(processed_images) != 1:
            processed_images.append(image.resize((image_size, image_size)))
        return processed_images

    def load_image(self, image: Image.Image, input_size: int = 448, max_num: int = 12):
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = torch.stack([transform(img) for img in images])
        return pixel_values

    def __init__(
        self,
        model_config: Dict[str, Any],
        device_manager,
        inference_config: Dict[str, Any],
    ):
        super().__init__(model_config, device_manager)
        self.inference_config = inference_config

    def load_model(self) -> None:
        """Load InternVL 3.5 Instruct (GitHub format): AutoModel + AutoTokenizer, trust_remote_code."""
        model_id = self.model_config["model_id"]
        logger.info(f"Loading InternVL 3.5 Instruct model: {model_id}")
        try:
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                logger.info("Using HuggingFace token from environment")

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=False,
                token=hf_token,
            )

            model_kwargs = self.device_manager.get_model_kwargs()
            try:
                import flash_attn
                use_flash_attn = True
                logger.info("Flash Attention 2 is available, enabling it")
            except ImportError:
                use_flash_attn = False
                logger.warning("Flash Attention 2 not installed, using standard attention")

            _original_item = torch.Tensor.item
            def _safe_item(tensor):
                try:
                    dev = getattr(tensor, 'device', None)
                    if dev is not None and str(dev).startswith('meta'):
                        return 0
                except Exception:
                    pass
                return _original_item(tensor)

            with patch.object(torch.Tensor, 'item', _safe_item):
                self.model = AutoModel.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    low_cpu_mem_usage=True,
                    use_flash_attn=use_flash_attn,
                    trust_remote_code=True,
                    token=hf_token,
                    device_map=model_kwargs.get('device_map', 'auto'),
                )

            if not hasattr(self.model, 'all_tied_weights_keys'):
                self.model.all_tied_weights_keys = []
            if not hasattr(self.model, '_tied_weights_keys'):
                self.model._tied_weights_keys = []

            self.model.eval()

            if self.inference_config.get('use_torch_compile', False):
                compile_fn = getattr(torch, 'compile', None)
                if compile_fn is not None:
                    try:
                        self.model = compile_fn(self.model, mode='reduce-overhead')
                        logger.info("torch.compile enabled (mode=reduce-overhead)")
                    except Exception as e:
                        logger.warning(f"torch.compile failed: {e}")
                else:
                    logger.warning("use_torch_compile is true but torch.compile not available")

            try:
                model_dev = getattr(self.model, 'device', None) or next(self.model.parameters()).device
                if str(model_dev).startswith('cuda'):
                    logger.info(f"Model is on GPU: {model_dev}")
                else:
                    logger.warning(f"Model is on CPU: {model_dev}")
            except Exception as e:
                logger.warning(f"Could not determine model device: {e}")

            logger.info(f"Successfully loaded {self.model_config['name']}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def warmup(self) -> None:
        if self.model is None or self.tokenizer is None:
            return
        try:
            warmup_image = Image.new("RGB", (448, 448), color=(255, 255, 255))
            pixel_values = self.load_image(warmup_image, input_size=448, max_num=1)
            model_dev = getattr(self.model, 'device', None) or next(self.model.parameters()).device
            pixel_values = pixel_values.to(torch.bfloat16).to(model_dev)
            question = "<image>\nHi"
            generation_config = dict(max_new_tokens=5, do_sample=False)
            self.model.chat(self.tokenizer, pixel_values, question, generation_config)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("InternVL warmup done.")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal): {e}")

    def process_image(self, image: Image.Image, prompt: str = None) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if prompt is None:
            prompt = DEFAULT_OCR_PROMPT

        model_dev = getattr(self.model, 'device', None) or next(self.model.parameters()).device
        pixel_values = self.load_image(image, input_size=448, max_num=12)
        pixel_values = pixel_values.to(torch.bfloat16).to(model_dev)

        question = f'<image>\n{prompt}'
        generation_config = dict(
            max_new_tokens=self.inference_config.get('max_new_tokens', 2048),
            do_sample=True,
        )

        try:
            response = self.model.chat(
                self.tokenizer,
                pixel_values,
                question,
                generation_config,
            )
            if isinstance(response, tuple):
                output_text = response[0]
            else:
                output_text = str(response)
            if not output_text or not output_text.strip():
                raise ValueError("InternVL returned empty response")
        except Exception as e:
            logger.error(f"Error during InternVL chat: {e}")
            raise RuntimeError(f"InternVL failed to process image: {e}")
        finally:
            del pixel_values
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return self._clean_code_fences(output_text.strip())

    def process_batch(
        self,
        images: List[Image.Image],
        prompts: List[str] = None,
    ) -> List[str]:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if prompts is None:
            prompts = [DEFAULT_OCR_PROMPT] * len(images)
        if len(prompts) != len(images):
            raise ValueError("Number of prompts must match number of images")
        results = []
        for image, prompt in zip(images, prompts):
            try:
                results.append(self.process_image(image, prompt))
            except Exception as e:
                logger.error(f"Error processing image in batch: {e}")
                results.append(f"Error: {str(e)}")
        return results
