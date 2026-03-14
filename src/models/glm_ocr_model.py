"""
GLM-OCR model handler for document OCR.

Based on https://huggingface.co/zai-org/GLM-OCR
- Uses AutoProcessor + AutoModelForImageTextToText
- Document parsing prompt: "Text Recognition:" (also "Formula Recognition:", "Table Recognition:")
- 0.9B parameters, efficient inference
"""

from typing import Any, Dict, List, Optional
from PIL import Image
import torch
import tempfile
import os
import logging
from pathlib import Path

from transformers import AutoProcessor, AutoModelForImageTextToText
from dotenv import load_dotenv

from .base_model import BaseOCRModel

load_dotenv()
logger = logging.getLogger(__name__)

# Default prompt for document text extraction (per GLM-OCR model card)
DEFAULT_OCR_PROMPT = "Text Recognition:"


class GLMOCRModel(BaseOCRModel):
    """GLM-OCR model handler for document understanding and OCR."""

    def __init__(
        self,
        model_config: Dict[str, Any],
        device_manager,
        inference_config: Dict[str, Any],
    ):
        super().__init__(model_config, device_manager)
        self.inference_config = inference_config

    def load_model(self) -> None:
        """Load GLM-OCR processor and model (per Hugging Face model card)."""
        model_id = self.model_config["model_id"]
        logger.info(f"Loading GLM-OCR model: {model_id}")

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
            # GLM-OCR card: torch_dtype="auto", device_map="auto"
            dtype = model_kwargs.get("torch_dtype")
            if dtype is None:
                dtype = "auto"

            self.model = AutoModelForImageTextToText.from_pretrained(
                model_id,
                torch_dtype=dtype,
                device_map=model_kwargs.get("device_map", "auto"),
                trust_remote_code=True,
                token=hf_token,
            )

            self.model.eval()

            # Optional: torch.compile
            if self.inference_config.get("use_torch_compile", False):
                compile_fn = getattr(torch, "compile", None)
                if compile_fn is not None:
                    try:
                        self.model = compile_fn(self.model, mode="reduce-overhead")
                        logger.info("torch.compile enabled (mode=reduce-overhead)")
                    except Exception as e:
                        logger.warning(f"torch.compile failed: {e}")
                else:
                    logger.warning("use_torch_compile is true but torch.compile not available")

            try:
                model_dev = next(self.model.parameters()).device
                if str(model_dev).startswith("cuda"):
                    logger.info(f"Model is on GPU: {model_dev}")
                else:
                    logger.warning(f"Model is on CPU: {model_dev}")
            except Exception as e:
                logger.warning(f"Could not determine model device: {e}")

            logger.info(f"Successfully loaded {self.model_config['name']}")

        except Exception as e:
            logger.error(f"Failed to load GLM-OCR model: {e}")
            raise

    def warmup(self) -> None:
        """Minimal inference to trigger CUDA/compilation."""
        if self.model is None or self.processor is None:
            return
        try:
            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
                Image.new("RGB", (64, 64), color=(255, 255, 255)).save(f.name)
                path = f.name
            try:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image", "url": path},
                            {"type": "text", "text": DEFAULT_OCR_PROMPT},
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
                inputs.pop("token_type_ids", None)
                inputs = {k: v.to(self.model.device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
                with torch.no_grad():
                    self.model.generate(**inputs, max_new_tokens=5)
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                logger.info("GLM-OCR warmup done.")
            finally:
                os.unlink(path)
        except Exception as e:
            logger.warning(f"GLM-OCR warmup failed (non-fatal): {e}")

    def process_image(self, image: Image.Image, prompt: str = None) -> str:
        """Run GLM-OCR on a single image (per model card Transformers example)."""
        if self.model is None or self.processor is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        if prompt is None or not prompt.strip():
            prompt = DEFAULT_OCR_PROMPT
        else:
            # User prompt is appended after the task prefix if needed; GLM-OCR supports
            # "Text Recognition:", "Formula Recognition:", "Table Recognition:" or custom.
            pass

        # GLM-OCR processor expects image via "url" (file path); save PIL to temp file
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
            if image.mode != "RGB":
                image = image.convert("RGB")
            image.save(f.name)
            image_path = f.name

        try:
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "url": image_path},
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
            inputs.pop("token_type_ids", None)
            device = next(self.model.parameters()).device
            if hasattr(inputs, "to"):
                inputs = inputs.to(device)
            else:
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            with torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=self.inference_config.get("max_new_tokens", 8192),
                )

            # Decode only the generated part (per model card)
            input_ids = inputs["input_ids"] if isinstance(inputs, dict) else getattr(inputs, "input_ids", inputs)
            input_len = input_ids.shape[1]
            output_ids = generated_ids[0][input_len:]
            output_text = self.processor.decode(output_ids, skip_special_tokens=False)

            del inputs
            del generated_ids
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            return output_text.strip()
        finally:
            os.unlink(image_path)

    def process_batch(
        self,
        images: List[Image.Image],
        prompts: List[str] = None,
    ) -> List[str]:
        """Process multiple images (one call per image)."""
        if prompts is None:
            prompts = [DEFAULT_OCR_PROMPT] * len(images)
        if len(prompts) != len(images):
            raise ValueError("Number of prompts must match number of images")
        return [self.process_image(img, pr) for img, pr in zip(images, prompts)]
