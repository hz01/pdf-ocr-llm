"""
OCRFlux-3B model handler (ChatDOC).

Based on Qwen2.5-VL-3B-Instruct; outputs clean Markdown.
https://huggingface.co/ChatDOC/OCRFlux-3B
"""

from typing import Any, Dict, List, Optional
from PIL import Image
import torch
from transformers import AutoProcessor
import logging
import os
from dotenv import load_dotenv

from .base_model import BaseOCRModel

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_OCR_PROMPT = (
    "Extract all text from this image. Preserve the layout, formatting, tables, and structure. "
    "Use Markdown syntax for headings, lists, tables, bold, italic, etc. "
    "Output only the formatted text without code blocks or fences."
)


class OCRFluxModel(BaseOCRModel):
    """OCRFlux-3B (ChatDOC) — Qwen2.5-VL based, markdown output."""

    @staticmethod
    def _clean_code_fences(text: str) -> str:
        import re
        text = re.sub(r"^```(?:markdown)?\s*\n", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n```\s*$", "", text, flags=re.MULTILINE)
        return text.strip()

    def __init__(
        self,
        model_config: Dict[str, Any],
        device_manager,
        inference_config: Dict[str, Any],
    ):
        super().__init__(model_config, device_manager)
        self.inference_config = inference_config
        self.max_pixels = model_config.get("max_pixels", 1280)
        self.min_pixels = model_config.get("min_pixels", 256)

    def load_model(self) -> None:
        """Load OCRFlux-3B (Qwen2.5-VL) processor and model."""
        model_id = self.model_config["model_id"]
        logger.info(f"Loading OCRFlux model: {model_id}")

        try:
            hf_token = os.getenv("HF_TOKEN")
            if hf_token:
                logger.info("Using HuggingFace token from environment")

            self.processor = AutoProcessor.from_pretrained(
                model_id,
                trust_remote_code=True,
                token=hf_token,
            )

            model_kwargs = self.device_manager.get_model_kwargs()
            try:
                import flash_attn
                model_kwargs["attn_implementation"] = "flash_attention_2"
                logger.info("Flash Attention 2 enabled for OCRFlux")
            except ImportError:
                pass

            try:
                from transformers import Qwen2_5_VLForConditionalGeneration
                self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    token=hf_token,
                    **model_kwargs,
                )
            except ImportError:
                from transformers import AutoModelForImageTextToText
                self.model = AutoModelForImageTextToText.from_pretrained(
                    model_id,
                    trust_remote_code=True,
                    token=hf_token,
                    **model_kwargs,
                )

            self.model.eval()

            if self.inference_config.get("use_torch_compile", False):
                compile_fn = getattr(torch, "compile", None)
                if compile_fn:
                    try:
                        self.model = compile_fn(self.model, mode="reduce-overhead")
                        logger.info("torch.compile enabled for OCRFlux")
                    except Exception as e:
                        logger.warning(f"torch.compile failed: {e}")

            try:
                model_dev = next(self.model.parameters()).device
                if str(model_dev).startswith("cuda"):
                    logger.info(f"OCRFlux model on GPU: {model_dev}")
                else:
                    logger.warning(f"OCRFlux model on CPU: {model_dev}")
            except Exception:
                pass

            logger.info(f"Successfully loaded {self.model_config['name']}")
        except Exception as e:
            logger.error(f"Failed to load OCRFlux model: {e}")
            raise

    def warmup(self) -> None:
        if self.model is None or self.processor is None:
            return
        try:
            img = Image.new("RGB", (224, 224), color=(255, 255, 255))
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": "Hi"},
                    ],
                }
            ]
            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            dev = next(self.model.parameters()).device
            inputs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
            with torch.no_grad():
                self.model.generate(**inputs, max_new_tokens=5)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            logger.info("OCRFlux warmup done.")
        except Exception as e:
            logger.warning(f"OCRFlux warmup failed (non-fatal): {e}")

    def process_image(self, image: Image.Image, prompt: str = None) -> str:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if prompt is None or not prompt.strip():
            prompt = DEFAULT_OCR_PROMPT

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        inputs = self.processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        )
        dev = next(self.model.parameters()).device
        inputs = {k: v.to(dev) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        max_new_tokens = self.inference_config.get("max_new_tokens", 2048)
        with torch.no_grad():
            generated_ids = self.model.generate(**inputs, max_new_tokens=max_new_tokens)

        input_ids = inputs["input_ids"] if isinstance(inputs, dict) else getattr(inputs, "input_ids", inputs)
        gen_ids_trimmed = [out_ids[len(in_seq):] for in_seq, out_ids in zip(input_ids, generated_ids)]
        output_text = self.processor.batch_decode(
            gen_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]

        del inputs
        del generated_ids
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return self._clean_code_fences(output_text.strip())

    def process_batch(
        self,
        images: List[Image.Image],
        prompts: List[str] = None,
    ) -> List[str]:
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")
        if prompts is None:
            prompts = [DEFAULT_OCR_PROMPT] * len(images)
        if len(prompts) != len(images):
            raise ValueError("Number of prompts must match number of images")
        return [self.process_image(img, pr) for img, pr in zip(images, prompts)]
