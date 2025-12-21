## Notebook Name: Chessmate Data Prep.ipynb

---

## 1. Introduction
eta amar  GambitFlow/Nexus-Core ei model er dataset bananor notebook. 

---

### Cell 1
```python

# Cell 1: Final One-Click Training and Export

# --- ১. পুরো পরিবেশ সেটআপ (যদি রিস্টার্টের পর কিছু মিসিং থাকে) ---
# PyTorch 2.5.1 + onnxscript + onnx
print("⚙️ Final Environment Setup...")
# torch-2.5.1 ইন্সটল নিশ্চিত করা
!pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# onnx, onnxscript (CRITICAL FIX)
!pip install onnx onnxscript huggingface_hub

import torch
import torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import numpy as np
import sqlite3
import json
import time
import os
import threading
import random
from huggingface_hub import hf_hub_download, HfApi

# --- ২. গ্লোবাল কনফিগারেশন ---
if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    raise RuntimeError("❌ No GPU Found.")

# আপনার তথ্য
HF_TOKEN = "MY-HF-TOKEN"
HF_USERNAME = "Rafs-an09002"
REPO_ID = f"{HF_USERNAME}/chessmate-data-v2"
MODEL_REPO_ID = f"{HF_USERNAME}/chessmate-model-v2"
FILENAME = "chess_stats_v2.db"


# --- ৩. ডেটা ডাউনলোড (Hugging Face) ---
print("\n⬇️ Downloading Dataset...")
db_path = hf_hub_download(repo_id=REPO_ID, filename=FILENAME, repo_type="dataset", local_dir="/content/data")
print(f"✅ Download Complete: {db_path}")


# --- ৪. মডেল আর্কিটেকচার (ResNet) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

class ChessResNet(nn.Module):
    def __init__(self, num_residual_blocks=10, num_filters=128):
        super().__init__()
        self.conv_in = nn.Sequential(
            nn.Conv2d(12, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.residual_blocks(x)
        value = self.value_head(x)
        return torch.tanh(value)

# --- ৫. ডেটা/টেনসর লজিক (আগের মতো) ---
def fen_to_tensor(fen):
    position = fen.split(' ')[0]
    tensor = np.zeros((12, 8, 8), dtype=np.float32)
    piece_to_channel = {'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5, 'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11}
    rank = 0; file = 0
    for char in position:
        if char == '/': rank += 1; file = 0
        elif char.isdigit(): file += int(char)
        elif char in piece_to_channel:
            tensor[piece_to_channel[char], rank, file] = 1.0
            file += 1
    return torch.from_numpy(tensor)

class SQLiteIterableDataset(IterableDataset):
    def __init__(self, db_path, shuffle_buffer_size=50000):
        self.db_path = db_path
        self.shuffle_buffer_size = shuffle_buffer_size
    def __iter__(self):
        conn = sqlite3.connect(self.db_path); cursor = conn.cursor()
        cursor.execute("SELECT fen, stats FROM positions")
        buffer = []
        for row in cursor:
            fen, stats_json = row; stats = json.loads(stats_json)
            total = stats['total']
            if total < 50: continue

            # (Simplified Score)
            white_wins = stats['moves'].get('e4', {'white':0}).get('white', 0)
            black_wins = stats['moves'].get('e4', {'black':0}).get('black', 0)
            score = (white_wins - black_wins) / total

            buffer.append({'fen': fen, 'score': score})
            if len(buffer) >= self.shuffle_buffer_size:
                yield self.process_sample(buffer.pop(random.randrange(len(buffer))))
        while buffer:
            yield self.process_sample(buffer.pop(random.randrange(len(buffer))))
        conn.close()
    def process_sample(self, sample):
        tensor = fen_to_tensor(sample['fen']); score = torch.tensor([sample['score']], dtype=torch.float32)
        return tensor.squeeze(0), score

# --- ৬. ট্রেনিং ইনিশিয়ালাইজেশন ---
train_dataset = SQLiteIterableDataset(db_path, shuffle_buffer_size=100000)
train_loader = DataLoader(train_dataset, batch_size=512, num_workers=0)

model = ChessResNet(num_residual_blocks=10, num_filters=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
criterion = nn.MSELoss()
EPOCHS = 3 # ৩ ইপক (টেস্টের জন্য)
scaler = torch.amp.GradScaler(device="cuda")

# --- ৭. ট্রেনিং লুপ ---
print("\n🧠 Starting PRO-LEVEL Training...")
for epoch in range(EPOCHS):
    model.train(); running_loss = 0.0; total_batches = 0
    optimizer.zero_grad()
    for i, (inputs_cpu, targets_cpu) in enumerate(train_loader):
        inputs = inputs_cpu.to(device); targets = targets_cpu.to(device)
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs); loss = criterion(outputs, targets)
        scaler.scale(loss / 2).backward()
        if (i + 1) % 2 == 0:
            scaler.step(optimizer); scaler.update(); optimizer.zero_grad()
        running_loss += loss.item(); total_batches += 1
        if total_batches % 500 == 0:
            print(f"Epoch {epoch+1}/{EPOCHS} | Batch {i+1} | Loss: {running_loss/500:.6f}")
            running_loss = 0.0

# --- ৮. ONNX এক্সপোর্ট ---
print("\n💾 Finalizing Model & Exporting to ONNX...")
model.eval(); onnx_path = os.path.join("/content/data", "chess_model_v2.onnx")
model.to('cpu'); dummy_input = torch.randn(1, 12, 8, 8).to('cpu')

try:
    torch.onnx.export(model, dummy_input, onnx_path, export_params=True, opset_version=17, do_constant_folding=True, input_names=['board_state'], output_names=['evaluation'])

    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    if file_size > 5:
        print(f"✅ ONNX Export Successful! Size: {file_size:.2f} MB")

        # Hugging Face Upload
        api = HfApi(token=HF_TOKEN)
        api.upload_file(path_or_fileobj=onnx_path, path_in_repo="chess_model_v2.onnx", repo_id=MODEL_REPO_ID, repo_type="model")
        print(f"\n🎉 Model Upload Complete! URL: https://huggingface.co/models/{MODEL_REPO_ID}/resolve/main/chess_model_v2.onnx")
    else:
        print(f"❌ Failed: File too small ({file_size:.2f} MB).")

except Exception as e:
    print(f"❌ FINAL EXPORT FAILED: {e}")
```


Output:



```text
⚙️ Final Environment Setup...
Looking in indexes: https://download.pytorch.org/whl/cu124
Collecting torch==2.5.1
  Downloading https://download.pytorch.org/whl/cu124/torch-2.5.1%2Bcu124-cp312-cp312-linux_x86_64.whl (908.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 908.2/908.2 MB 1.6 MB/s eta 0:00:00
Collecting torchvision==0.20.1
  Downloading https://download.pytorch.org/whl/cu124/torchvision-0.20.1%2Bcu124-cp312-cp312-linux_x86_64.whl (7.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 7.3/7.3 MB 105.9 MB/s eta 0:00:00
Collecting torchaudio==2.5.1
  Downloading https://download.pytorch.org/whl/cu124/torchaudio-2.5.1%2Bcu124-cp312-cp312-linux_x86_64.whl (3.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.4/3.4 MB 120.6 MB/s eta 0:00:00
Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (3.20.0)
Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (4.15.0)
Requirement already satisfied: networkx in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (3.6.1)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (3.1.6)
Requirement already satisfied: fsspec in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (2025.3.0)
Collecting nvidia-cuda-nvrtc-cu12==12.4.127 (from torch==2.5.1)
  Downloading https://download.pytorch.org/whl/cu124/nvidia_cuda_nvrtc_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (24.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 24.6/24.6 MB 99.9 MB/s eta 0:00:00
Collecting nvidia-cuda-runtime-cu12==12.4.127 (from torch==2.5.1)
  Downloading https://download.pytorch.org/whl/cu124/nvidia_cuda_runtime_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (883 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 883.7/883.7 kB 64.8 MB/s eta 0:00:00
Collecting nvidia-cuda-cupti-cu12==12.4.127 (from torch==2.5.1)
  Downloading https://download.pytorch.org/whl/cu124/nvidia_cuda_cupti_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (13.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 13.8/13.8 MB 131.2 MB/s eta 0:00:00
Collecting nvidia-cudnn-cu12==9.1.0.70 (from torch==2.5.1)
  Downloading https://download.pytorch.org/whl/cu124/nvidia_cudnn_cu12-9.1.0.70-py3-none-manylinux2014_x86_64.whl (664.8 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 664.8/664.8 MB 2.0 MB/s eta 0:00:00
Collecting nvidia-cublas-cu12==12.4.5.8 (from torch==2.5.1)
  Downloading https://download.pytorch.org/whl/cu124/nvidia_cublas_cu12-12.4.5.8-py3-none-manylinux2014_x86_64.whl (363.4 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 363.4/363.4 MB 4.0 MB/s eta 0:00:00
Collecting nvidia-cufft-cu12==11.2.1.3 (from torch==2.5.1)
  Downloading https://download.pytorch.org/whl/cu124/nvidia_cufft_cu12-11.2.1.3-py3-none-manylinux2014_x86_64.whl (211.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 211.5/211.5 MB 4.9 MB/s eta 0:00:00
Collecting nvidia-curand-cu12==10.3.5.147 (from torch==2.5.1)
  Downloading https://download.pytorch.org/whl/cu124/nvidia_curand_cu12-10.3.5.147-py3-none-manylinux2014_x86_64.whl (56.3 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 56.3/56.3 MB 15.6 MB/s eta 0:00:00
Collecting nvidia-cusolver-cu12==11.6.1.9 (from torch==2.5.1)
  Downloading https://download.pytorch.org/whl/cu124/nvidia_cusolver_cu12-11.6.1.9-py3-none-manylinux2014_x86_64.whl (127.9 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 127.9/127.9 MB 7.5 MB/s eta 0:00:00
Collecting nvidia-cusparse-cu12==12.3.1.170 (from torch==2.5.1)
  Downloading https://download.pytorch.org/whl/cu124/nvidia_cusparse_cu12-12.3.1.170-py3-none-manylinux2014_x86_64.whl (207.5 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 207.5/207.5 MB 5.5 MB/s eta 0:00:00
Collecting nvidia-nccl-cu12==2.21.5 (from torch==2.5.1)
  Downloading https://download.pytorch.org/whl/nvidia_nccl_cu12-2.21.5-py3-none-manylinux2014_x86_64.whl (188.7 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 188.7/188.7 MB 6.0 MB/s eta 0:00:00
Collecting nvidia-nvtx-cu12==12.4.127 (from torch==2.5.1)
  Downloading https://download.pytorch.org/whl/cu124/nvidia_nvtx_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (99 kB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 99.1/99.1 kB 10.1 MB/s eta 0:00:00
Collecting nvidia-nvjitlink-cu12==12.4.127 (from torch==2.5.1)
  Downloading https://download.pytorch.org/whl/cu124/nvidia_nvjitlink_cu12-12.4.127-py3-none-manylinux2014_x86_64.whl (21.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 21.1/21.1 MB 104.8 MB/s eta 0:00:00
Collecting triton==3.1.0 (from torch==2.5.1)
  Downloading https://download.pytorch.org/whl/triton-3.1.0-cp312-cp312-manylinux_2_17_x86_64.manylinux2014_x86_64.whl (209.6 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 209.6/209.6 MB 5.4 MB/s eta 0:00:00
Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (75.2.0)
Collecting sympy==1.13.1 (from torch==2.5.1)
  Downloading https://download.pytorch.org/whl/sympy-1.13.1-py3-none-any.whl (6.2 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.2/6.2 MB 106.2 MB/s eta 0:00:00
Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from torchvision==0.20.1) (2.0.2)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.12/dist-packages (from torchvision==0.20.1) (11.3.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy==1.13.1->torch==2.5.1) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch==2.5.1) (3.0.3)
Installing collected packages: triton, sympy, nvidia-nvtx-cu12, nvidia-nvjitlink-cu12, nvidia-nccl-cu12, nvidia-curand-cu12, nvidia-cufft-cu12, nvidia-cuda-runtime-cu12, nvidia-cuda-nvrtc-cu12, nvidia-cuda-cupti-cu12, nvidia-cublas-cu12, nvidia-cusparse-cu12, nvidia-cudnn-cu12, nvidia-cusolver-cu12, torch, torchvision, torchaudio
  Attempting uninstall: triton
    Found existing installation: triton 3.5.0
    Uninstalling triton-3.5.0:
      Successfully uninstalled triton-3.5.0
  Attempting uninstall: sympy
    Found existing installation: sympy 1.14.0
    Uninstalling sympy-1.14.0:
      Successfully uninstalled sympy-1.14.0
  Attempting uninstall: nvidia-nvtx-cu12
    Found existing installation: nvidia-nvtx-cu12 12.6.77
    Uninstalling nvidia-nvtx-cu12-12.6.77:
      Successfully uninstalled nvidia-nvtx-cu12-12.6.77
  Attempting uninstall: nvidia-nvjitlink-cu12
    Found existing installation: nvidia-nvjitlink-cu12 12.6.85
    Uninstalling nvidia-nvjitlink-cu12-12.6.85:
      Successfully uninstalled nvidia-nvjitlink-cu12-12.6.85
  Attempting uninstall: nvidia-nccl-cu12
    Found existing installation: nvidia-nccl-cu12 2.27.5
    Uninstalling nvidia-nccl-cu12-2.27.5:
      Successfully uninstalled nvidia-nccl-cu12-2.27.5
  Attempting uninstall: nvidia-curand-cu12
    Found existing installation: nvidia-curand-cu12 10.3.7.77
    Uninstalling nvidia-curand-cu12-10.3.7.77:
      Successfully uninstalled nvidia-curand-cu12-10.3.7.77
  Attempting uninstall: nvidia-cufft-cu12
    Found existing installation: nvidia-cufft-cu12 11.3.0.4
    Uninstalling nvidia-cufft-cu12-11.3.0.4:
      Successfully uninstalled nvidia-cufft-cu12-11.3.0.4
  Attempting uninstall: nvidia-cuda-runtime-cu12
    Found existing installation: nvidia-cuda-runtime-cu12 12.6.77
    Uninstalling nvidia-cuda-runtime-cu12-12.6.77:
      Successfully uninstalled nvidia-cuda-runtime-cu12-12.6.77
  Attempting uninstall: nvidia-cuda-nvrtc-cu12
    Found existing installation: nvidia-cuda-nvrtc-cu12 12.6.77
    Uninstalling nvidia-cuda-nvrtc-cu12-12.6.77:
      Successfully uninstalled nvidia-cuda-nvrtc-cu12-12.6.77
  Attempting uninstall: nvidia-cuda-cupti-cu12
    Found existing installation: nvidia-cuda-cupti-cu12 12.6.80
    Uninstalling nvidia-cuda-cupti-cu12-12.6.80:
      Successfully uninstalled nvidia-cuda-cupti-cu12-12.6.80
  Attempting uninstall: nvidia-cublas-cu12
    Found existing installation: nvidia-cublas-cu12 12.6.4.1
    Uninstalling nvidia-cublas-cu12-12.6.4.1:
      Successfully uninstalled nvidia-cublas-cu12-12.6.4.1
  Attempting uninstall: nvidia-cusparse-cu12
    Found existing installation: nvidia-cusparse-cu12 12.5.4.2
    Uninstalling nvidia-cusparse-cu12-12.5.4.2:
      Successfully uninstalled nvidia-cusparse-cu12-12.5.4.2
  Attempting uninstall: nvidia-cudnn-cu12
    Found existing installation: nvidia-cudnn-cu12 9.10.2.21
    Uninstalling nvidia-cudnn-cu12-9.10.2.21:
      Successfully uninstalled nvidia-cudnn-cu12-9.10.2.21
  Attempting uninstall: nvidia-cusolver-cu12
    Found existing installation: nvidia-cusolver-cu12 11.7.1.2
    Uninstalling nvidia-cusolver-cu12-11.7.1.2:
      Successfully uninstalled nvidia-cusolver-cu12-11.7.1.2
  Attempting uninstall: torch
    Found existing installation: torch 2.9.0+cu126
    Uninstalling torch-2.9.0+cu126:
      Successfully uninstalled torch-2.9.0+cu126
  Attempting uninstall: torchvision
    Found existing installation: torchvision 0.24.0+cu126
    Uninstalling torchvision-0.24.0+cu126:
      Successfully uninstalled torchvision-0.24.0+cu126
  Attempting uninstall: torchaudio
    Found existing installation: torchaudio 2.9.0+cu126
    Uninstalling torchaudio-2.9.0+cu126:
      Successfully uninstalled torchaudio-2.9.0+cu126
Successfully installed nvidia-cublas-cu12-12.4.5.8 nvidia-cuda-cupti-cu12-12.4.127 nvidia-cuda-nvrtc-cu12-12.4.127 nvidia-cuda-runtime-cu12-12.4.127 nvidia-cudnn-cu12-9.1.0.70 nvidia-cufft-cu12-11.2.1.3 nvidia-curand-cu12-10.3.5.147 nvidia-cusolver-cu12-11.6.1.9 nvidia-cusparse-cu12-12.3.1.170 nvidia-nccl-cu12-2.21.5 nvidia-nvjitlink-cu12-12.4.127 nvidia-nvtx-cu12-12.4.127 sympy-1.13.1 torch-2.5.1+cu124 torchaudio-2.5.1+cu124 torchvision-0.20.1+cu124 triton-3.1.0
Collecting onnx
  Downloading onnx-1.20.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
Collecting onnxscript
  Downloading onnxscript-0.5.6-py3-none-any.whl.metadata (13 kB)
Requirement already satisfied: huggingface_hub in /usr/local/lib/python3.12/dist-packages (0.36.0)
Requirement already satisfied: numpy>=1.23.2 in /usr/local/lib/python3.12/dist-packages (from onnx) (2.0.2)
Requirement already satisfied: protobuf>=4.25.1 in /usr/local/lib/python3.12/dist-packages (from onnx) (5.29.5)
Requirement already satisfied: typing_extensions>=4.7.1 in /usr/local/lib/python3.12/dist-packages (from onnx) (4.15.0)
Requirement already satisfied: ml_dtypes>=0.5.0 in /usr/local/lib/python3.12/dist-packages (from onnx) (0.5.4)
Collecting onnx_ir<2,>=0.1.12 (from onnxscript)
  Downloading onnx_ir-0.1.12-py3-none-any.whl.metadata (3.2 kB)
Requirement already satisfied: packaging in /usr/local/lib/python3.12/dist-packages (from onnxscript) (25.0)
Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (3.20.0)
Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (2025.3.0)
Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (6.0.3)
Requirement already satisfied: requests in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (2.32.4)
Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (4.67.1)
Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (1.2.0)
Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface_hub) (3.4.4)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface_hub) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface_hub) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface_hub) (2025.11.12)
Downloading onnx-1.20.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (18.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.1/18.1 MB 116.8 MB/s eta 0:00:00
Downloading onnxscript-0.5.6-py3-none-any.whl (683 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 683.0/683.0 kB 55.2 MB/s eta 0:00:00
Downloading onnx_ir-0.1.12-py3-none-any.whl (129 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 129.3/129.3 kB 13.6 MB/s eta 0:00:00
Installing collected packages: onnx, onnx_ir, onnxscript
Successfully installed onnx-1.20.0 onnx_ir-0.1.12 onnxscript-0.5.6

⬇️ Downloading Dataset...
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
✅ Download Complete: /content/data/chess_stats_v2.db

🧠 Starting PRO-LEVEL Training...

💾 Finalizing Model & Exporting to ONNX...
✅ ONNX Export Successful! Size: 13.34 MB
Processing Files (1 / 1)      : 100%
 14.0MB / 14.0MB, 10.0MB/s  
New Data Upload               : 100%
 14.0MB / 14.0MB, 10.0MB/s  
  .../data/chess_model_v2.onnx: 100%
 14.0MB / 14.0MB            

🎉 Model Upload Complete! URL: https://huggingface.co/models/Rafs-an09002/chessmate-model-v2/resolve/main/chess_model_v2.onnx
```

---

### Cell 2
```python
# Cell 2: ResNet Model Definition
import torch.nn.init as init

# --- ১. ResNet ব্লক ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU()
        self._initialize_weights()

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

    # Kaiming He Initialization (Advanced Fix)
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)


# --- ২. ResNet মডেল ---
class ChessResNet(nn.Module):
    def __init__(self, num_residual_blocks=10, num_filters=128):
        super().__init__()

        # Initial Block: (12, 8, 8) -> (128, 8, 8)
        self.conv_in = nn.Sequential(
            nn.Conv2d(12, num_filters, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_filters),
            nn.ReLU()
        )

        # Residual Tower
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_residual_blocks)]
        )

        # Value Head (Output: Evaluation Score)
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1) # Final Evaluation Score
        )

    def forward(self, x):
        x = self.conv_in(x)
        x = self.residual_blocks(x)
        value = self.value_head(x)
        return torch.tanh(value) # [-1, 1] Range

print("✅ ResNet Model Architecture Defined. Ready for Training (Cell 3).")

```


Output:



```text
✅ ResNet Model Architecture Defined. Ready for Training (Cell 3).
```

---

### Cell 3
```python

# Cell 3: ResNet Training, Mixed Precision, and ONNX Export (Final Fixes)

# --- ১. ট্রেনিং সেটআপ ---
model = ChessResNet(num_residual_blocks=10, num_filters=128).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
criterion = nn.MSELoss()
EPOCHS = 5
GRADIENT_ACCUMULATION_STEPS = 2

# CRITICAL FIX: torch.amp ব্যবহার করা
scaler = torch.amp.GradScaler(device="cuda")

# --- ২. ট্রেনিং লুপ ---
print(f"\n🧠 Starting PRO-LEVEL Training on {device}...")
print(f"Total Epochs: {EPOCHS} | Effective Batch Size: {512 * GRADIENT_ACCUMULATION_STEPS}")

start_time = time.time()
total_batches = 0

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    optimizer.zero_grad()

    for i, (inputs_cpu, targets_cpu) in enumerate(train_loader):

        # CRITICAL FIX: ডেটা লোড হওয়ার সাথে সাথে GPU তে পাঠানো
        inputs = inputs_cpu.to(device)
        targets = targets_cpu.to(device)

        # Mixed Precision Context
        with torch.amp.autocast(device_type="cuda", dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

        # গ্রাডিয়েন্ট স্কেলিং ও ব্যাকপ্রোপাগেশন
        scaler.scale(loss / GRADIENT_ACCUMULATION_STEPS).backward()

        # গ্রাডিয়েন্ট অ্যাকুমুলেশন
        if (i + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item()
        total_batches += 1

        if total_batches % 500 == 0: # প্রতি ৫০০টি ব্যাচে আপডেট
            print(f"Epoch {epoch+1}/{EPOCHS} | Step {i+1} | Loss: {running_loss/500:.6f}")
            running_loss = 0.0

    # ইপক শেষে অবশিষ্ট গ্রাডিয়েন্ট কমিট
    if (i + 1) % GRADIENT_ACCUMULATION_STEPS != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()

    print(f"\n🎉 EPOCH {epoch+1} COMPLETE! Total Time: {(time.time() - start_time):.2f}s")


# --- ৩. ONNX এক্সপোর্ট এবং Hugging Face আপলোড ---
# (আগের মতো - ফিক্সড)

```


Output:



```text


🧠 Starting PRO-LEVEL Training on cuda...
Total Epochs: 5 | Effective Batch Size: 1024

🎉 EPOCH 1 COMPLETE! Total Time: 15.72s

🎉 EPOCH 2 COMPLETE! Total Time: 26.61s

🎉 EPOCH 3 COMPLETE! Total Time: 37.43s

🎉 EPOCH 4 COMPLETE! Total Time: 47.51s

🎉 EPOCH 5 COMPLETE! Total Time: 57.82s
```

---

### Cell 4
```python

# Cell 4: ONNX Export and Hugging Face Upload (Final Verified)

from huggingface_hub import HfApi
import torch
import os
import time

# --- ১. ONNX এক্সপোর্ট ---
print("\n💾 Finalizing Model & Exporting to ONNX...")
model.eval()

# ফাইল পাথ
onnx_path = os.path.join("/content/data", "chess_model_v2.onnx")

# এক্সপোর্টের জন্য মডেলে ও টেনসরকে CPU তে পাঠানো
model.to('cpu')
dummy_input = torch.randn(1, 12, 8, 8).to('cpu')

try:
    # এই সেটিংস (opset 17, do_constant_folding=True) v2.5.1-এ কাজ করবে
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True, # মডেল ট্রেইন হওয়ার পর এটি অন রাখা নিরাপদ
        input_names=['board_state'],
        output_names=['evaluation'],
        dynamic_axes=None
    )

    # ২. ফাইল সাইজ চেক (CRITICAL CHECK)
    file_size = os.path.getsize(onnx_path) / (1024 * 1024)

    if file_size < 5:
        print(f"❌ FATAL ERROR: Model Size is only {file_size:.2f} MB! Something went wrong.")
        raise RuntimeError("Model export failed to save weights.")

    print(f"✅ ONNX Export Successful! Verified Size: {file_size:.2f} MB")

except Exception as e:
    print(f"❌ ONNX Export Failed: {e}")
    raise e


# --- ৩. Hugging Face মডেল রিপোজিটরিতে আপলোড ---
HF_TOKEN = "MY-HF-TOKEN"
HF_USERNAME = "Rafs-an09002"
REPO_ID = f"{HF_USERNAME}/chessmate-model-v2" # v2 মডেল রিপোজিটরি

api = HfApi(token=HF_TOKEN)

print(f"\n🚀 Uploading Model to: {REPO_ID}...")

try:
    api.create_repo(repo_id=REPO_ID, repo_type="model", exist_ok=True)

    # আসল ফাইলটি আপলোড করা হলো
    api.upload_file(
        path_or_fileobj=onnx_path,
        path_in_repo="chess_model_v2.onnx",
        repo_id=REPO_ID,
        repo_type="model"
    )

    print("\n🎉 V2 Model Upload Complete!")
    print(f"🔗 URL: https://huggingface.co/models/{REPO_ID}/resolve/main/chess_model_v2.onnx")

except Exception as e:
    print(f"\n❌ Model Upload Failed: {e}")
```


Output:



```text


💾 Finalizing Model & Exporting to ONNX...
✅ ONNX Export Successful! Verified Size: 13.34 MB

🚀 Uploading Model to: Rafs-an09002/chessmate-model-v2...
Processing Files (1 / 1)      : 100%
 14.0MB / 14.0MB, 10.0MB/s  
New Data Upload               : 100%
 14.0MB / 14.0MB, 10.0MB/s  
  .../data/chess_model_v2.onnx: 100%
 14.0MB / 14.0MB            

🎉 V2 Model Upload Complete!
🔗 URL: https://huggingface.co/models/Rafs-an09002/chessmate-model-v2/resolve/main/chess_model_v2.onnx
```
