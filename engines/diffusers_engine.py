from __future__ import annotations

import gc
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config import Settings
from engines.base import ImageEngine


class DiffusersEngine(ImageEngine):
    name = "diffusers"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipe: Any | None = None
        self._torch: Any | None = None
        self._device = "NVIDIA CUDA"

    def load(self) -> None:
        if self._pipe is not None:
            return
        try:
            import torch
            from diffusers import BitsAndBytesConfig as DiffusersBitsAndBytesConfig
            from diffusers import Flux2KleinPipeline
            from diffusers.quantizers import PipelineQuantizationConfig
            from transformers import BitsAndBytesConfig as TransformersBitsAndBytesConfig
        except ImportError as exc:
            raise RuntimeError("Diffusers backend dependencies are not installed.") from exc
        if not torch.cuda.is_available():
            if torch.version.cuda is None:
                raise RuntimeError(
                    "CUDA is not available. This PyTorch installation is CPU-only. "
                    "Install the CUDA PyTorch wheel documented in README.md."
                )
            raise RuntimeError("CUDA is not available. Windows benchmark requires an NVIDIA CUDA GPU.")

        self._torch = torch
        self._device = torch.cuda.get_device_name(0)
        dtype = torch.float16  # GTX 1070 does not support BF16 efficiently.
        kwargs: dict[str, Any] = {"torch_dtype": dtype, "low_cpu_mem_usage": True}
        if self.settings.windows_quantization != "none":
            use_4bit = self.settings.windows_quantization == "4bit"
            # Flux.2 Klein has a Diffusers transformer and a Transformers Qwen3
            # text encoder. Each needs its respective BitsAndBytes config type.
            kwargs["quantization_config"] = PipelineQuantizationConfig(
                quant_mapping={
                    "transformer": DiffusersBitsAndBytesConfig(
                        load_in_4bit=use_4bit,
                        load_in_8bit=not use_4bit,
                        bnb_4bit_compute_dtype=dtype,
                    ),
                    "text_encoder": TransformersBitsAndBytesConfig(
                        load_in_4bit=use_4bit,
                        load_in_8bit=not use_4bit,
                        bnb_4bit_compute_dtype=dtype,
                    ),
                }
            )
        try:
            self._pipe = Flux2KleinPipeline.from_pretrained(self.settings.model_id, **kwargs)
            self._configure_memory_strategy()
            self._pipe.set_progress_bar_config(disable=True)
        except Exception as exc:
            self._pipe = None
            raise RuntimeError(f"Diffusers model initialization failed: {exc}") from exc

    def _configure_memory_strategy(self) -> None:
        assert self._pipe is not None
        if self.settings.windows_cpu_offload:
            # Sequential offload re-materializes components from ``meta`` tensors
            # and is incompatible with this BitsAndBytes-quantized FLUX.2 setup.
            # Model offload still keeps one component at a time on the GTX 1070,
            # while supporting the quantized transformer/text encoder.
            self._pipe.enable_model_cpu_offload()
        else:
            self._pipe.to("cuda")
        try:
            self._pipe.enable_attention_slicing()
        except Exception:
            pass

    def generate(
        self,
        prompt: str,
        reference_images: list[str] | None,
        width: int,
        height: int,
        seed: int,
    ) -> dict[str, Any]:
        self.load()
        assert self._pipe is not None and self._torch is not None
        actual_seed = random.SystemRandom().randint(0, 2**31 - 1) if seed == -1 else seed
        self._torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        try:
            kwargs: dict[str, Any] = {
                "prompt": prompt, "height": height, "width": width,
                "num_inference_steps": self.settings.steps, "guidance_scale": 1.0,
                "generator": self._torch.Generator("cpu").manual_seed(actual_seed),
            }
            if reference_images:
                from PIL import Image

                kwargs["image"] = [Image.open(path).convert("RGB") for path in reference_images]
            image = self._pipe(**kwargs).images[0]
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                self._torch.cuda.empty_cache()
                gc.collect()
                raise RuntimeError(
                    "CUDA out of memory. Try 512x512 or enable the configured low-VRAM mode."
                ) from exc
            raise
        generation_seconds = time.perf_counter() - started
        output_path = self._output_path(actual_seed)
        image.save(output_path)
        gpu_memory_mb = round(self._torch.cuda.max_memory_allocated() / 1024 / 1024, 1)
        return {
            "image_path": str(output_path), "backend": self.name, "device": self._device,
            "model": "flux2-klein-4b", "seed": actual_seed, "width": width, "height": height,
            "generation_seconds": generation_seconds, "total_seconds": time.perf_counter() - started,
            "memory_info": {"system_ram_mb": None, "gpu_memory_mb": gpu_memory_mb},
        }

    def _output_path(self, seed: int) -> Path:
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.settings.output_dir / f"{stamp}_{self.name}_{seed}.png"

    def info(self) -> dict[str, str]:
        return {
            "platform": "Windows", "backend": self.name, "device": self._device,
            "model": "FLUX.2 Klein 4B", "memory_mode": self.settings.windows_memory_mode,
        }
