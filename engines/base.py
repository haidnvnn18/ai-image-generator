from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ImageEngine(ABC):
    """The only inference contract consumed by the UI layer."""

    name: str

    @abstractmethod
    def load(self) -> None:
        """Load model resources once and keep them warm."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        reference_images: list[str] | None,
        width: int,
        height: int,
        seed: int,
    ) -> dict[str, Any]:
        """Generate and return the shared result contract."""

    @abstractmethod
    def info(self) -> dict[str, str]:
        """Return safe, human-readable backend metadata for the UI."""
