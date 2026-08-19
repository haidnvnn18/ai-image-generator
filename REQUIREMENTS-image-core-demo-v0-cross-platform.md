# REQUIREMENTS — AI Image Core Demo V0 (Mac + Windows)

## 1. Mục tiêu

Xây một demo tối giản để kiểm tra trực tiếp **chất lượng ảnh và tốc độ inference** trên cả:

- **Mac mini M1 16GB**
- **Windows PC với CPU AMD + NVIDIA GTX 1070 8GB**

Demo phải dùng **cùng một UI, cùng input contract, cùng benchmark format**, nhưng cho phép mỗi hệ điều hành dùng inference backend phù hợp với phần cứng.

Mục tiêu duy nhất của V0:

- Upload 0..N ảnh tham chiếu.
- Nhập prompt.
- Chạy inference bằng backend phù hợp:
  - macOS Apple Silicon → `mflux + MLX`
  - Windows NVIDIA → `Diffusers + PyTorch CUDA`
- Model family mục tiêu: **FLUX.2 Klein 4B**
- Hiển thị ảnh output.
- Hiển thị thời gian xử lý.
- Ghi log benchmark để so sánh Mac và Windows.

V0 KHÔNG phải production server.

V0 KHÔNG cần:

- Redis
- FastAPI riêng
- distributed queue
- worker/server tách riêng
- database
- auth
- Docker
- multi-machine orchestration
- object storage
- WebSocket
- scheduler

Mục tiêu là trả lời các câu hỏi:

1. Chất lượng FLUX.2 Klein 4B có đủ tốt cho use case thực tế không?
2. Mac mini M1 16GB mất bao lâu để generate một ảnh?
3. GTX 1070 8GB có chạy được model với cấu hình phù hợp không?
4. Output giữa 2 backend có chất lượng tương đương ở mức chấp nhận được không?
5. Nền tảng nào phù hợp hơn cho bước tiếp theo?

---

## 2. Nguyên tắc kiến trúc

Cross-platform ở **tầng interface**, không ép cả Mac và Windows dùng cùng runtime.

```text
                    Browser
                       │
                       ▼
                   Gradio UI
                       │
                       ▼
                  ImageEngine
                       │
             ┌─────────┴─────────┐
             │                   │
             ▼                   ▼
        macOS / M1          Windows / NVIDIA
             │                   │
       MfluxEngine        DiffusersEngine
             │                   │
           mflux              Diffusers
             │                   │
            MLX              PyTorch CUDA
             │                   │
      Apple Metal GPU        GTX 1070
```

UI và benchmark code không được phụ thuộc trực tiếp vào `mflux` hoặc `diffusers`.

Inference backend phải được encapsulate trong thư mục `engines/`.

---

## 3. Tech stack

### Shared

| Thành phần | Công nghệ |
|---|---|
| Language | Python 3.11+ |
| UI | Gradio |
| Image IO | Pillow |
| Benchmark log | CSV |
| Config | `.env` hoặc environment variables |

### macOS backend

| Thành phần | Công nghệ |
|---|---|
| Runtime | mflux |
| Compute | MLX / Metal |
| Target | Apple Silicon |
| Model | FLUX.2 Klein 4B |
| Quantization | 4-bit mặc định |

### Windows backend

| Thành phần | Công nghệ |
|---|---|
| Runtime | Hugging Face Diffusers |
| Compute | PyTorch CUDA |
| Target GPU | NVIDIA GTX 1070 8GB |
| Model | FLUX.2 Klein 4B |
| Memory strategy | quantization và/hoặc CPU offload khi cần |

Không dùng PyTorch MPS trên Mac trong V0.

Mac phải ưu tiên `mflux + MLX`.

---

## 4. Cấu trúc thư mục

```text
image-core-demo/
├── app.py
├── config.py
├── benchmark.py
├── engines/
│   ├── __init__.py
│   ├── base.py
│   ├── mflux_engine.py
│   └── diffusers_engine.py
├── requirements-common.txt
├── requirements-mac.txt
├── requirements-windows.txt
├── README.md
├── outputs/
│   └── .gitkeep
├── logs/
│   └── benchmark.csv
└── .gitignore
```

Không tạo Redis/server/worker abstraction trong V0.

---

## 5. Engine interface

File:

```text
engines/base.py
```

Định nghĩa interface chung:

```python
from abc import ABC, abstractmethod


class ImageEngine(ABC):
    name: str

    @abstractmethod
    def load(self) -> None:
        ...

    @abstractmethod
    def generate(
        self,
        prompt: str,
        reference_images: list[str] | None,
        width: int,
        height: int,
        seed: int,
    ) -> dict:
        ...
```

`app.py` chỉ được gọi interface này.

Không được chứa logic `mflux` hoặc `diffusers` trực tiếp trong UI layer.

---

## 6. Chọn backend

Backend được chọn theo environment variable:

```text
IMAGE_BACKEND=auto
```

Các giá trị hợp lệ:

```text
auto
mflux
diffusers
```

### auto

Nếu:

```text
platform == macOS
AND Apple Silicon
```

→ dùng:

```text
mflux
```

Nếu:

```text
platform == Windows
AND CUDA available
```

→ dùng:

```text
diffusers
```

Nếu không match:

```text
raise RuntimeError
```

Không tự fallback xuống CPU cho image generation benchmark vì CPU không phản ánh mục tiêu performance.

### Explicit mode

Cho phép:

```bash
IMAGE_BACKEND=mflux
```

hoặc:

```bash
IMAGE_BACKEND=diffusers
```

để debug.

---

## 7. Model configuration

Model family cố định:

```text
FLUX.2 Klein 4B
```

### Mac

Default:

```text
backend = mflux
quantization = 4-bit
steps = 4
```

### Windows GTX 1070 8GB

Default strategy:

```text
backend = diffusers
steps = 4
memory mode = low_vram
```

`DiffusersEngine` phải hỗ trợ cấu hình memory-saving thông qua config/env.

Ví dụ:

```text
WINDOWS_QUANTIZATION=4bit
WINDOWS_CPU_OFFLOAD=true
```

Không hard-code rằng model full precision sẽ fit GTX 1070 8GB.

Nếu full model không fit VRAM:

- sử dụng quantized weights nếu implementation hỗ trợ
- hoặc CPU offload
- hoặc combination phù hợp
- nếu vẫn không chạy được thì trả error rõ ràng

Không silently đổi sang model khác.

---

## 8. Model consistency

Cả Mac và Windows phải dùng:

```text
same model family = FLUX.2 Klein 4B
same prompt
same seed
same step count
same resolution
```

Không yêu cầu output giống nhau pixel-by-pixel.

Do backend khác nhau:

```text
mflux / MLX
vs
Diffusers / PyTorch CUDA
```

nên cùng seed vẫn có thể tạo output khác nhau.

Benchmark cần so:

- quality
- latency
- memory
- stability

Không kiểm tra pixel equality.

---

## 9. UI

UI chỉ cần:

### 9.1 Backend info

Hiển thị read-only:

```text
Platform
Backend
Device
Model
Quantization / memory mode
```

Ví dụ Mac:

```text
Platform: macOS
Backend: mflux
Device: Apple M1
Model: FLUX.2 Klein 4B
Quantization: 4-bit
```

Ví dụ Windows:

```text
Platform: Windows
Backend: diffusers
Device: NVIDIA GeForce GTX 1070
Model: FLUX.2 Klein 4B
Memory mode: 4-bit + CPU offload
```

---

### 9.2 Reference Images

Cho phép upload:

```text
0..N images
```

Định dạng:

- JPEG
- PNG
- WebP

Nếu 0 ảnh:

```text
text-to-image
```

Nếu có ảnh:

```text
image-conditioned generation/edit
```

Nếu backend hiện tại chưa hỗ trợ reference image hoặc multi-image:

- báo lỗi rõ ràng
- không ignore input
- không fallback silently sang text-to-image

---

### 9.3 Prompt

Textarea bắt buộc.

Nếu prompt rỗng:

```text
Prompt is required.
```

---

### 9.4 Resolution

Dropdown:

```text
512 x 512
640 x 640
704 x 704
768 x 768
```

Default:

```text
512 x 512
```

Lý do chọn 512 default:

- an toàn hơn cho GTX 1070 8GB
- phù hợp benchmark baseline
- sau đó mới tăng resolution

---

### 9.5 Seed

Cho phép:

```text
-1 = random
>= 0 = fixed seed
```

Default:

```text
42
```

Benchmark nên ưu tiên fixed seed.

---

### 9.6 Generate Button

Button:

```text
Generate
```

Khi đang inference:

- chỉ cho phép 1 job tại một thời điểm
- disable hoặc queue click mới
- hiển thị `Generating...`

Không chạy concurrent inference trong V0.

---

### 9.7 Output

Hiển thị:

```text
Output Image
Backend
Device
Actual Seed
Resolution
Generation Time
Total Time
Output Path
Error
```

---

## 10. MfluxEngine

File:

```text
engines/mflux_engine.py
```

Trách nhiệm:

- load FLUX.2 Klein 4B qua mflux
- quantization 4-bit
- giữ model warm
- text-to-image
- image-conditioned generation nếu mflux version hỗ trợ
- trả result theo contract chung

Model load:

```text
app start
   ↓
MfluxEngine.load()
   ↓
model stays resident
   ↓
UI launch
```

Không load model mỗi request.

---

## 11. DiffusersEngine

File:

```text
engines/diffusers_engine.py
```

Trách nhiệm:

- load FLUX.2 Klein 4B qua Diffusers
- chạy PyTorch CUDA
- hỗ trợ GTX 1070 8GB
- dùng memory-saving strategy theo config
- giữ pipeline warm nếu có thể
- text-to-image
- image-conditioned generation nếu pipeline/backend hỗ trợ
- trả result theo contract chung

### GTX 1070 rules

GTX 1070 là test hardware constrained.

Không assume:

```text
full BF16 model fits VRAM
```

Engine phải log:

```text
CUDA device
VRAM total
selected memory mode
quantization mode
CPU offload enabled/disabled
```

Nếu OOM:

```text
Generation failed: CUDA out of memory.
Try lower resolution or enable low-VRAM mode.
```

Không crash toàn app.

---

## 12. Generate result contract

Cả 2 backend phải trả cùng format:

```python
{
    "image_path": "...",
    "backend": "mflux",
    "device": "Apple M1",
    "model": "flux2-klein-4b",
    "seed": 42,
    "width": 512,
    "height": 512,
    "generation_seconds": 20.12,
    "total_seconds": 20.55,
    "memory_info": {
        "system_ram_mb": None,
        "gpu_memory_mb": None
    }
}
```

Các field memory có thể `None` nếu backend không lấy được số liệu chính xác.

Không được fail generation chỉ vì không đọc được memory metric.

---

## 13. Output storage

Output lưu:

```text
./outputs/
```

Tên file phải bao gồm platform/backend để dễ compare:

```text
YYYYMMDD_HHMMSS_<backend>_<seed>.png
```

Ví dụ:

```text
20260819_091530_mflux_42.png
20260819_091815_diffusers_42.png
```

Không overwrite file cũ.

---

## 14. Benchmark logging

Mỗi generation append vào:

```text
logs/benchmark.csv
```

Columns:

```text
timestamp
platform
backend
device
model
memory_mode
mode
reference_count
prompt
width
height
steps
seed
generation_seconds
total_seconds
success
error
output_path
```

Ví dụ:

```csv
timestamp,platform,backend,device,model,memory_mode,mode,reference_count,prompt,width,height,steps,seed,generation_seconds,total_seconds,success,error,output_path
2026-08-19T09:20:01,macos,mflux,Apple M1,flux2-klein-4b,q4,text2img,0,"cinematic portrait",512,512,4,42,18.4,18.8,true,,outputs/xxx.png
```

Windows:

```csv
2026-08-19T09:25:01,windows,diffusers,GTX 1070,flux2-klein-4b,q4_cpu_offload,text2img,0,"cinematic portrait",512,512,4,42,55.2,56.1,true,,outputs/yyy.png
```

---

## 15. Benchmark procedure

Cùng một bộ test chạy trên cả Mac và Windows.

### 15.1 Prompt set

Chuẩn bị tối thiểu:

1. portrait
2. product photography
3. landscape/interior
4. complex scene
5. full-body human
6. text/sign rendering

---

### 15.2 Resolution matrix

Chạy:

```text
512
640
704
768
```

Nếu Windows GTX 1070 fail/OOM ở resolution nào:

- record failure
- không cố ép app crash
- tiếp tục benchmark resolution thấp hơn

---

### 15.3 Seed

Dùng:

```text
42
123
999
```

---

### 15.4 Warm run

Generation đầu tiên:

```text
warm-up
```

Không dùng để tính benchmark chính.

Từ lần thứ 2 trở đi mới tính latency.

---

## 16. Comparison report

Tạo script:

```text
benchmark.py
```

Có khả năng đọc:

```text
logs/benchmark.csv
```

và in summary:

```text
Platform
Backend
Resolution
Success Count
Fail Count
Average Generation Time
p50
p95
```

Ví dụ:

```text
Mac M1 / mflux / 512
avg: 18.9s
p50: 18.4s
p95: 21.2s

GTX 1070 / diffusers / 512
avg: 54.1s
p50: 52.8s
p95: 61.4s
```

Không cần chart trong V0.

---

## 17. Dependency strategy

Không dùng một `requirements.txt` chung cho cả 2 platform nếu dependency conflict.

Dùng:

```text
requirements-common.txt
requirements-mac.txt
requirements-windows.txt
```

### requirements-common.txt

Tối thiểu:

```text
gradio
Pillow
python-dotenv
```

### requirements-mac.txt

Bao gồm:

```text
-r requirements-common.txt
mflux
```

### requirements-windows.txt

Bao gồm:

```text
-r requirements-common.txt
torch
diffusers
transformers
accelerate
safetensors
```

Nếu 4-bit quantization implementation yêu cầu package bổ sung như `bitsandbytes`, thêm vào Windows requirements sau khi xác nhận environment chạy được.

Không pin version ngẫu nhiên ngay từ đầu.

Sau khi có một environment chạy thành công:

```text
freeze/pin versions
```

để benchmark reproducible.

---

## 18. README

README phải có 2 phần riêng.

# macOS / Apple Silicon

Flow:

```bash
python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements-mac.txt

IMAGE_BACKEND=mflux python app.py
```

Nếu project dùng `uv` để cài mflux thì README được phép dùng `uv`, nhưng command phải copy/paste được.

---

# Windows / NVIDIA

Flow PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -r requirements-windows.txt

$env:IMAGE_BACKEND="diffusers"
python app.py
```

README phải có bước verify CUDA:

```python
import torch
print(torch.cuda.is_available())
print(torch.cuda.get_device_name(0))
```

Expected trên máy test:

```text
True
NVIDIA GeForce GTX 1070
```

Nếu CUDA unavailable:

- dừng
- không benchmark CPU fallback

---

## 19. Error handling

### Prompt empty

```text
Prompt is required.
```

### Unsupported image

```text
Unsupported image format.
```

### Mac model load fail

```text
Mflux model initialization failed: <reason>
```

### Windows CUDA unavailable

```text
CUDA is not available.
Windows benchmark requires an NVIDIA CUDA GPU.
```

### Windows OOM

```text
CUDA out of memory.
Try 512x512 or enable the configured low-VRAM mode.
```

### Unsupported image-conditioned mode

```text
This backend does not currently support reference-image generation in this configuration.
```

Không silently bỏ qua reference images.

---

## 20. Concurrency

V0:

```text
max inference concurrency = 1
```

Áp dụng cả Mac và Windows.

Không test 2 concurrent generations trong V0.

Mục tiêu hiện tại là:

```text
single-worker latency
quality
memory
stability
```

---

## 21. Acceptance criteria

V0 hoàn thành khi:

### Shared

- [ ] `python app.py` mở được Gradio UI.
- [ ] UI code không phụ thuộc trực tiếp vào mflux/diffusers.
- [ ] Backend được chọn qua `IMAGE_BACKEND`.
- [ ] Có text-to-image.
- [ ] Có upload reference image.
- [ ] Nếu backend không hỗ trợ reference image thì báo rõ.
- [ ] Có resolution 512 / 640 / 704 / 768.
- [ ] Có seed cố định và random.
- [ ] Có generation time và total time.
- [ ] Output lưu xuống `outputs/`.
- [ ] Benchmark append vào CSV.
- [ ] Generation fail không làm app chết.
- [ ] Chỉ chạy 1 inference cùng lúc.

### Mac

- [ ] Chạy được trên Mac mini M1 16GB.
- [ ] Sử dụng mflux/MLX.
- [ ] FLUX.2 Klein 4B load một lần.
- [ ] Quantization 4-bit.
- [ ] Generate được ít nhất 512x512.

### Windows

- [ ] Chạy được trên Windows với CPU AMD + GTX 1070 8GB.
- [ ] CUDA được detect.
- [ ] Sử dụng Diffusers/PyTorch CUDA.
- [ ] Có low-VRAM strategy.
- [ ] Generate được ít nhất 512x512 hoặc trả failure rõ ràng nếu model configuration không fit.
- [ ] Không fallback CPU silently.

### Benchmark

- [ ] Cùng prompt/seed/resolution chạy được trên cả hai platform nếu hardware cho phép.
- [ ] `benchmark.py` tính được avg/p50/p95.
- [ ] Có thể compare Mac M1 và GTX 1070 từ cùng CSV schema.

---

## 22. Không nằm trong scope V0

Không implement:

- Redis
- central API server
- distributed worker
- worker heartbeat
- scheduler
- multi-machine
- Docker
- auth
- rate limit
- S3/MinIO
- webhook
- persistent job history
- model switching trên UI
- LoRA
- upscale
- face swap
- background removal
- object removal
- SAM
- moderation
- billing

---

## 23. Chuẩn bị cho V1

V0 phải giữ separation:

```text
Gradio UI
    │
ImageEngine interface
    │
    ├── MfluxEngine
    └── DiffusersEngine
```

Sau này V1 có thể reuse hai engine trong worker process:

```text
Server
  │
Queue
  │
Worker
  │
ImageEngine
  │
  ├── mflux
  └── diffusers
```

Nhưng KHÔNG implement server/queue trong V0.

---

## 24. Definition of Done

Developer hoàn tất V0 khi có thể thực hiện:

### Mac

1. Clone project.
2. Install Mac dependencies.
3. Run app.
4. Generate ảnh bằng mflux.
5. Test 512/640/704/768.
6. Record latency.

### Windows

1. Clone cùng project.
2. Install Windows dependencies.
3. Verify CUDA GTX 1070.
4. Run cùng app.
5. Generate cùng prompt/seed.
6. Record latency.
7. Nếu VRAM không đủ ở resolution cao, app fail graceful và benchmark record failure.

### Cuối cùng

Từ:

```text
logs/benchmark.csv
```

có thể trả lời:

```text
Mac M1:
- quality?
- p50?
- p95?
- max stable resolution?

GTX 1070:
- chạy được không?
- quality?
- p50?
- p95?
- max stable resolution?
```

Sau V0 mới quyết định:

```text
A. tiếp tục build distributed worker system

hoặc

B. thay model/runtime/hardware strategy
```
