from __future__ import annotations

import platform
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from config import Settings
from engines.base import ImageEngine


class MfluxEngine(ImageEngine):
    name = "mflux"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._model: Any | None = None
        self._edit_model: Any | None = None
        self._device = platform.processor() or "Apple Silicon"

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            from mflux import Flux2

            self._model = Flux2(
                model=self.settings.mac_model_id,
                base_model="flux2-klein-4b",
            )
        except Exception as exc:
            raise RuntimeError(f"Mflux model initialization failed: {exc}") from exc

    def _get_edit_model(self) -> Any:
        """Create the mflux edit pipeline only when reference mode is first used."""
        if self._edit_model is not None:
            return self._edit_model
        try:
            from mflux.models.common.config import ModelConfig
            from mflux.models.flux2.variants import Flux2KleinEdit

            self._edit_model = Flux2KleinEdit(
                model_config=ModelConfig.flux2_klein_4b(),
                model_path=self.settings.mac_model_id,
            )
            return self._edit_model
        except Exception as exc:
            raise RuntimeError(
                "This backend does not currently support reference-image generation "
                f"in this configuration. {exc}"
            ) from exc

    def generate(
        self,
        prompt: str,
        reference_images: list[str] | None,
        width: int,
        height: int,
        seed: int,
    ) -> dict[str, Any]:
        self.load()
        actual_seed = random.SystemRandom().randint(0, 2**31 - 1) if seed == -1 else seed
        started = time.perf_counter()
        if reference_images:
            model = self._get_edit_model()
            image = model.generate_image(
                seed=actual_seed,
                prompt=prompt,
                image_paths=reference_images,
                num_inference_steps=self.settings.steps,
                width=width,
                height=height,
            )
        else:
            image = self._model.generate_image(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=self.settings.steps,
                seed=actual_seed,
            )
        generation_seconds = time.perf_counter() - started
        output_path = self._output_path(actual_seed)
        image.save(output_path)
        return {
            "image_path": str(output_path), "backend": self.name, "device": self._device,
            "model": "flux2-klein-4b", "seed": actual_seed, "width": width, "height": height,
            "generation_seconds": generation_seconds, "total_seconds": time.perf_counter() - started,
            "memory_info": {"system_ram_mb": None, "gpu_memory_mb": None},
        }

    def _output_path(self, seed: int) -> Path:
        self.settings.output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        return self.settings.output_dir / f"{stamp}_{self.name}_{seed}.png"

    def info(self) -> dict[str, str]:
        return {
            "platform": "macOS", "backend": self.name, "device": self._device,
            "model": "FLUX.2 Klein 4B", "memory_mode": self.settings.mac_quantization,
        }
