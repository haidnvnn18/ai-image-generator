"""Engine factory. Runtime-specific imports remain inside the engine modules."""

from __future__ import annotations

from config import Settings, select_backend
from engines.base import ImageEngine


def create_engine(settings: Settings) -> ImageEngine:
    backend = select_backend(settings.backend)
    if backend == "mflux":
        from engines.mflux_engine import MfluxEngine

        return MfluxEngine(settings)
    if backend == "diffusers":
        from engines.diffusers_engine import DiffusersEngine

        return DiffusersEngine(settings)
    raise AssertionError(f"Unexpected backend: {backend}")
