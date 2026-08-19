from __future__ import annotations

import csv
import platform
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import gradio as gr

from config import get_settings
from engines import create_engine
from engines.base import ImageEngine


SETTINGS = get_settings()
INFERENCE_LOCK = threading.Lock()
ALLOWED_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
CSV_COLUMNS = [
    "timestamp", "platform", "backend", "device", "model", "memory_mode", "mode",
    "reference_count", "prompt", "width", "height", "steps", "seed",
    "generation_seconds", "total_seconds", "success", "error", "output_path",
]


def append_benchmark(
    engine: ImageEngine,
    prompt: str,
    reference_count: int,
    width: int,
    height: int,
    seed: int,
    result: dict[str, Any] | None = None,
    error: str = "",
    total_seconds: float | None = None,
) -> None:
    SETTINGS.log_file.parent.mkdir(parents=True, exist_ok=True)
    info = engine.info()
    is_success = result is not None
    row = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": info["platform"].lower(), "backend": info["backend"],
        "device": info["device"], "model": "flux2-klein-4b",
        "memory_mode": info["memory_mode"],
        "mode": "image_conditioned" if reference_count else "text2img",
        "reference_count": reference_count, "prompt": prompt, "width": width, "height": height,
        "steps": SETTINGS.steps, "seed": result["seed"] if result else seed,
        "generation_seconds": result["generation_seconds"] if result else "",
        "total_seconds": result["total_seconds"] if result else (total_seconds or ""),
        "success": str(is_success).lower(), "error": error, "output_path": result["image_path"] if result else "",
    }
    write_header = not SETTINGS.log_file.exists() or SETTINGS.log_file.stat().st_size == 0
    with SETTINGS.log_file.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def parse_resolution(value: str) -> tuple[int, int]:
    width, height = value.lower().replace(" ", "").split("x", maxsplit=1)
    return int(width), int(height)


def validate_reference_paths(files: list[str] | None) -> list[str]:
    paths = [str(item) for item in (files or [])]
    unsupported = [path for path in paths if Path(path).suffix.lower() not in ALLOWED_SUFFIXES]
    if unsupported:
        raise ValueError("Unsupported image format.")
    return paths


def render_info(engine: ImageEngine) -> str:
    info = engine.info()
    memory_label = "Quantization" if info["backend"] == "mflux" else "Memory mode"
    return (
        f"**Platform:** {info['platform']}  \n"
        f"**Backend:** {info['backend']}  \n"
        f"**Device:** {info['device']}  \n"
        f"**Model:** {info['model']}  \n"
        f"**{memory_label}:** {info['memory_mode']}"
    )


def render_result(result: dict[str, Any]) -> str:
    memory = result["memory_info"]
    gpu_memory = memory["gpu_memory_mb"]
    gpu_line = f"  \n**Peak GPU memory:** {gpu_memory} MB" if gpu_memory is not None else ""
    return (
        f"**Backend:** {result['backend']}  \n**Device:** {result['device']}  \n"
        f"**Actual seed:** {result['seed']}  \n**Resolution:** {result['width']}x{result['height']}  \n"
        f"**Generation time:** {result['generation_seconds']:.2f}s  \n"
        f"**Total time:** {result['total_seconds']:.2f}s  \n"
        f"**Output path:** `{result['image_path']}`{gpu_line}"
    )


def generate(
    engine: ImageEngine,
    prompt: str,
    files: list[str] | None,
    resolution: str,
    seed: float,
) -> tuple[str | None, str]:
    prompt = (prompt or "").strip()
    if not prompt:
        return None, "**Error:** Prompt is required."
    width, height = parse_resolution(resolution)
    try:
        reference_paths = validate_reference_paths(files)
    except ValueError as exc:
        return None, f"**Error:** {exc}"
    requested_seed = int(seed)
    started = time.perf_counter()
    with INFERENCE_LOCK:
        try:
            result = engine.generate(prompt, reference_paths or None, width, height, requested_seed)
            append_benchmark(engine, prompt, len(reference_paths), width, height, requested_seed, result=result)
            return result["image_path"], render_result(result)
        except Exception as exc:
            message = str(exc)
            append_benchmark(
                engine, prompt, len(reference_paths), width, height, requested_seed,
                error=message, total_seconds=time.perf_counter() - started,
            )
            return None, f"**Error:** {message}"


def build_demo(engine: ImageEngine) -> gr.Blocks:
    with gr.Blocks(title="AI Image Core Demo V0") as demo:
        gr.Markdown("# AI Image Core Demo V0\nCross-platform FLUX.2 Klein 4B benchmark harness.")
        gr.Markdown(render_info(engine))
        with gr.Row():
            with gr.Column():
                reference_images = gr.File(
                    label="Reference Images (optional)", file_count="multiple", type="filepath",
                    file_types=["image"],
                )
                prompt = gr.Textbox(label="Prompt", lines=5, placeholder="Describe the image to generate...")
                resolution = gr.Dropdown(
                    label="Resolution", choices=["512 x 512", "640 x 640", "704 x 704", "768 x 768"],
                    value="512 x 512",
                )
                seed = gr.Number(label="Seed (-1 = random)", value=42, precision=0)
                generate_button = gr.Button("Generate", variant="primary")
            with gr.Column():
                output_image = gr.Image(label="Output Image", type="filepath")
                output_info = gr.Markdown("Ready.")
        generate_button.click(
            fn=lambda p, f, r, s: generate(engine, p, f, r, s),
            inputs=[prompt, reference_images, resolution, seed],
            outputs=[output_image, output_info],
            show_progress="full",
        )
    return demo.queue(default_concurrency_limit=1, max_size=1)


def main() -> None:
    engine = create_engine(SETTINGS)
    print(f"Loading {engine.name} model once before launching the UI...")
    engine.load()
    demo = build_demo(engine)
    demo.launch()


if __name__ == "__main__":
    main()
