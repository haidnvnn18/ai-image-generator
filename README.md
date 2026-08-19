# AI Image Core Demo V0

Minimal, single-process Gradio benchmark harness for **FLUX.2 Klein 4B**. The browser UI and CSV format are shared; the inference engine is chosen by platform:

- macOS Apple Silicon: `mflux` + MLX
- Windows NVIDIA: Diffusers + PyTorch CUDA

There is intentionally no API server, queue service, database, or CPU fallback. The model is loaded once before the UI starts and inference is limited to one queued request at a time.

## Configuration

Create an optional `.env` file in the project root:

```dotenv
# auto | mflux | diffusers
IMAGE_BACKEND=auto
INFERENCE_STEPS=4

# Apple Silicon, 4-bit mflux checkpoint
MFLUX_MODEL_ID=RunPod/FLUX.2-klein-4B-mflux-4bit

# NVIDIA / Diffusers checkpoint
MODEL_ID=black-forest-labs/FLUX.2-klein-4B
WINDOWS_QUANTIZATION=4bit
WINDOWS_CPU_OFFLOAD=true
```

`WINDOWS_QUANTIZATION` accepts `4bit` (default), `8bit`, or `none`. For 4/8-bit runs the Diffusers engine applies BitsAndBytes quantization to both the Flux transformer and Qwen text encoder; a BitsAndBytes compatibility error is reported clearly and never falls back to full precision. `WINDOWS_CPU_OFFLOAD=true` enables model CPU offload to make GTX 1070 testing more feasible, at a substantial latency cost.

Both model repositories may require accepting their Hugging Face terms and authenticating first:

```bash
huggingface-cli login
```

## macOS / Apple Silicon

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-mac.txt
IMAGE_BACKEND=mflux python app.py
```

`auto` chooses mflux only on Apple Silicon. The default Mac model is a 4-bit mflux checkpoint; set `MFLUX_MODEL_ID` if you use a different compatible checkpoint.

## Windows / NVIDIA

Use Python 3.11+ and install the CUDA-enabled PyTorch wheel appropriate for your NVIDIA driver. Then:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt

.\.venv\Scripts\python.exe -c "import sys, torch; print(sys.executable); print(torch.__version__); print(torch.version.cuda); print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0))"

$env:IMAGE_BACKEND="diffusers"
.\.venv\Scripts\python.exe app.py
```

Expected verification result on the target machine:

```text
True
NVIDIA GeForce GTX 1070
```

The Python executable printed by the verification command must be `.venv\Scripts\python.exe`. If it reports a version ending in `+cpu` or `torch.version.cuda` is `None`, activate the project virtual environment and reinstall the CUDA wheel:

```powershell
.\.venv\Scripts\python.exe -m pip uninstall -y torch torchvision torchaudio
.\.venv\Scripts\python.exe -m pip install torch==2.13.0+cu126 --index-url https://download.pytorch.org/whl/cu126
.\.venv\Scripts\python.exe -m pip install -r requirements-windows.txt
```

If CUDA is still unavailable after this, stop there: this project intentionally does not benchmark CPU generation. A CUDA out-of-memory error is caught, written to the benchmark CSV, and displayed without terminating the app. Start GTX 1070 testing at 512×512.

## Use the UI

1. Optional: upload one or more JPEG, PNG, or WebP reference images.
2. Enter a prompt, choose 512/640/704/768, and set a seed (`-1` is random).
3. Click **Generate**. The result is written to `outputs/` using a timestamp/backend/seed filename.
4. Every successful or failed request is appended to `logs/benchmark.csv`.

Reference images are passed to the active engine. If the installed backend/model configuration cannot support image-conditioned generation, it returns a clear error rather than ignoring the images.

## Benchmark summary

Run after one or more generations:

```bash
python benchmark.py
```

The summary groups by platform, backend, device, and resolution, reporting success/failure counts plus average, p50, and p95 generation latency. For comparable Mac/Windows runs, keep prompt, seed, step count, and resolution identical. The first run is a warm-up; compare latency from run two onward.

## Project layout

```text
app.py                 Gradio UI; knows only ImageEngine
config.py              environment configuration and backend selection
benchmark.py           CSV summary utility
engines/base.py        shared engine contract
engines/mflux_engine.py
engines/diffusers_engine.py
outputs/               generated PNG files
logs/benchmark.csv     append-only benchmark log
```
