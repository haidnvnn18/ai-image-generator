"""Application configuration and deterministic backend selection."""

from __future__ import annotations

import os
import platform
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


ROOT_DIR = Path(__file__).resolve().parent
load_dotenv(ROOT_DIR / ".env")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


@dataclass(frozen=True)
class Settings:
    backend: str
    model_id: str
    mac_model_id: str
    steps: int
    mac_quantization: str
    windows_quantization: str
    windows_cpu_offload: bool
    output_dir: Path
    log_file: Path

    @property
    def windows_memory_mode(self) -> str:
        pieces = [self.windows_quantization]
        if self.windows_cpu_offload:
            pieces.append("cpu_offload")
        return "+".join(pieces)


def get_settings() -> Settings:
    settings = Settings(
        backend=os.getenv("IMAGE_BACKEND", "auto").strip().lower(),
        model_id=os.getenv("MODEL_ID", "black-forest-labs/FLUX.2-klein-4B"),
        mac_model_id=os.getenv("MFLUX_MODEL_ID", "RunPod/FLUX.2-klein-4B-mflux-4bit"),
        steps=int(os.getenv("INFERENCE_STEPS", "4")),
        mac_quantization=os.getenv("MAC_QUANTIZATION", "4-bit"),
        windows_quantization=os.getenv("WINDOWS_QUANTIZATION", "4bit").lower(),
        windows_cpu_offload=_env_bool("WINDOWS_CPU_OFFLOAD", True),
        output_dir=ROOT_DIR / "outputs",
        log_file=ROOT_DIR / "logs" / "benchmark.csv",
    )
    if settings.windows_quantization not in {"4bit", "8bit", "none"}:
        raise RuntimeError("WINDOWS_QUANTIZATION must be one of: 4bit, 8bit, none.")
    return settings


def select_backend(requested: str) -> str:
    """Select only an appropriate GPU runtime; never fall back to CPU."""
    if requested not in {"auto", "mflux", "diffusers"}:
        raise RuntimeError("IMAGE_BACKEND must be one of: auto, mflux, diffusers.")
    if requested != "auto":
        return requested

    system = platform.system()
    machine = platform.machine().lower()
    if system == "Darwin" and machine in {"arm64", "aarch64"}:
        return "mflux"
    if system == "Windows":
        try:
            import torch
        except ImportError as exc:
            raise RuntimeError("PyTorch is required to detect CUDA on Windows.") from exc
        if torch.cuda.is_available():
            return "diffusers"

    raise RuntimeError(
        "No supported image-generation backend detected. "
        "Auto mode requires Apple Silicon macOS or Windows with an NVIDIA CUDA GPU."
    )
