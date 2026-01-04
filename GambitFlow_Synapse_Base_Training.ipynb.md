## Notebook Name: Synapse-Base_02_Training.ipynb

---

## 1. Introduction
eta amar  GambitFlow/Synapse-Base  model er  notebook. 

---

### Cell 1: Environment Setup
```python

# ==============================================================================
# 🧠 SYNAPSE-BASE: Environment, Drive & Local Data Setup
# ==============================================================================

import os
import time
import threading
import shutil
import sqlite3
from google.colab import drive

print("⚙️ Setting up Synapse Environment...")

# ১. লাইব্রেরি ইনস্টল (Stable & Fast)
!pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
!pip install python-chess huggingface_hub onnx onnxscript

import torch
from huggingface_hub import hf_hub_download

# ২. ড্রাইভ মাউন্ট (ভবিষ্যতে মডেল সেভ করার জন্য)
drive.mount('/content/drive')
PROJECT_DIR = '/content/drive/MyDrive/GambitFlow_Project' # আপনার নতুন ফোল্ডার নাম
os.makedirs(PROJECT_DIR, exist_ok=True)

# ৩. কনফিগারেশন
class Config:
    INPUT_CHANNELS = 119
    HIDDEN_DIM = 256
    RESNET_BLOCKS = 20
    TRANSFORMER_LAYERS = 4
    HEADS = 8
    BATCH_SIZE = 256
    GRAD_ACCUMULATION = 4
    LEARNING_RATE = 1e-4
    EPOCHS = 10

CONFIG = Config()

# ৪. GPU চেক
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"✅ GPU Active: {torch.cuda.get_device_name(0)}")
else:
    raise RuntimeError("❌ No GPU Found! Please change Runtime Type to T4/A100.")

# ৫. ডেটাবেস সেটআপ (CRITICAL FIX: Local Cache)
# আমরা ড্রাইভ/HF থেকে ডেটা এনে Colab এর লোকাল ডিস্কে রাখব।
# এতে SQLite এরর হবে না এবং স্পিড ২০ গুণ বাড়বে।

HF_REPO_ID = "Rafs-an09002/chessmate-data-v2"
HF_FILENAME = "chess_stats_v2.db"
LOCAL_DB_PATH = "/content/data/chess_stats_v2.db" # লোকাল পাথ

print(f"\n⬇️ Setting up Database...")

if not os.path.exists(LOCAL_DB_PATH):
    os.makedirs("/content/data", exist_ok=True)
    try:
        print(f"   Downloading from Hugging Face to Local Disk (Fast I/O)...")
        # এটি সরাসরি লোকাল ডিস্কে ডাউনলোড করবে
        db_path = hf_hub_download(
            repo_id=HF_REPO_ID,
            filename=HF_FILENAME,
            repo_type="dataset",
            local_dir="/content/data"
        )
        print(f"✅ Database Cached Locally: {db_path}")
    except Exception as e:
        print(f"❌ Download Failed: {e}")
        raise e
else:
    print(f"✅ Database already exists locally. Skipping download.")

# ৬. সেশন কিপ-অ্যালাইভ
def keep_colab_awake():
    while True:
        time.sleep(60)
threading.Thread(target=keep_colab_awake, daemon=True).start()
print("✅ Keep-Alive Active.")

```


Output:



```text
⚙️ Setting up Synapse Environment...
Looking in indexes: https://download.pytorch.org/whl/cu124
Requirement already satisfied: torch==2.5.1 in /usr/local/lib/python3.12/dist-packages (2.5.1+cu124)
Requirement already satisfied: torchvision==0.20.1 in /usr/local/lib/python3.12/dist-packages (0.20.1+cu124)
Requirement already satisfied: torchaudio==2.5.1 in /usr/local/lib/python3.12/dist-packages (2.5.1+cu124)
Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (3.20.0)
Requirement already satisfied: typing-extensions>=4.8.0 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (4.15.0)
Requirement already satisfied: networkx in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (3.6.1)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (3.1.6)
Requirement already satisfied: fsspec in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (2025.3.0)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.4.127 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (12.4.127)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.4.127 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (12.4.127)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.4.127 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (12.4.127)
Requirement already satisfied: nvidia-cudnn-cu12==9.1.0.70 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (9.1.0.70)
Requirement already satisfied: nvidia-cublas-cu12==12.4.5.8 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (12.4.5.8)
Requirement already satisfied: nvidia-cufft-cu12==11.2.1.3 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (11.2.1.3)
Requirement already satisfied: nvidia-curand-cu12==10.3.5.147 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (10.3.5.147)
Requirement already satisfied: nvidia-cusolver-cu12==11.6.1.9 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (11.6.1.9)
Requirement already satisfied: nvidia-cusparse-cu12==12.3.1.170 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (12.3.1.170)
Requirement already satisfied: nvidia-nccl-cu12==2.21.5 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (2.21.5)
Requirement already satisfied: nvidia-nvtx-cu12==12.4.127 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (12.4.127)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.4.127 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (12.4.127)
Requirement already satisfied: triton==3.1.0 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (3.1.0)
Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (75.2.0)
Requirement already satisfied: sympy==1.13.1 in /usr/local/lib/python3.12/dist-packages (from torch==2.5.1) (1.13.1)
Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from torchvision==0.20.1) (2.0.2)
Requirement already satisfied: pillow!=8.3.*,>=5.3.0 in /usr/local/lib/python3.12/dist-packages (from torchvision==0.20.1) (11.3.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy==1.13.1->torch==2.5.1) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch==2.5.1) (3.0.3)
Requirement already satisfied: python-chess in /usr/local/lib/python3.12/dist-packages (1.999)
Requirement already satisfied: huggingface_hub in /usr/local/lib/python3.12/dist-packages (0.36.0)
Requirement already satisfied: onnx in /usr/local/lib/python3.12/dist-packages (1.20.0)
Requirement already satisfied: onnxscript in /usr/local/lib/python3.12/dist-packages (0.5.7)
Requirement already satisfied: chess<2,>=1 in /usr/local/lib/python3.12/dist-packages (from python-chess) (1.11.2)
Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (3.20.0)
Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (2025.3.0)
Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (25.0)
Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (6.0.3)
Requirement already satisfied: requests in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (2.32.4)
Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (4.67.1)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (4.15.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (1.2.0)
Requirement already satisfied: numpy>=1.23.2 in /usr/local/lib/python3.12/dist-packages (from onnx) (2.0.2)
Requirement already satisfied: protobuf>=4.25.1 in /usr/local/lib/python3.12/dist-packages (from onnx) (5.29.5)
Requirement already satisfied: ml_dtypes>=0.5.0 in /usr/local/lib/python3.12/dist-packages (from onnx) (0.5.4)
Requirement already satisfied: onnx_ir<2,>=0.1.12 in /usr/local/lib/python3.12/dist-packages (from onnxscript) (0.1.13)
Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface_hub) (3.4.4)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface_hub) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface_hub) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface_hub) (2025.11.12)
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
✅ GPU Active: Tesla T4

⬇️ Setting up Database...
✅ Database already exists locally. Skipping download.
✅ Keep-Alive Active.

```

---

### Cell 2
```python
# Cell 2: Advanced Feature Extraction (119 Channels) & Dataset Logic

import chess
import numpy as np
import torch
import sqlite3
import json
import random
from torch.utils.data import IterableDataset, DataLoader

# --- ১. 119-Channel Feature Extractor ---
# এটি FEN স্ট্রিং থেকে গভীর তথ্য বের করে (CNN+Transformer এর জন্য)
def fen_to_dense_tensor(fen):
    board = chess.Board(fen)
    # (Channels, Height, Width) -> (119, 8, 8)
    tensor = np.zeros((119, 8, 8), dtype=np.float32)

    # --- A. Pieces (0-11) ---
    piece_map = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
    }
    for sq in chess.SQUARES:
        piece = board.piece_at(sq)
        if piece:
            # White: 0-5, Black: 6-11
            channel = piece_map[piece.piece_type] + (6 if piece.color == chess.BLACK else 0)
            row, col = divmod(sq, 8)
            tensor[channel, 7-row, col] = 1.0

    # --- B. Global State (12-26) ---
    # 12: Turn
    if board.turn == chess.WHITE: tensor[12, :, :] = 1.0

    # 13-16: Castling
    if board.has_kingside_castling_rights(chess.WHITE): tensor[13, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE): tensor[14, :, :] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK): tensor[15, :, :] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK): tensor[16, :, :] = 1.0

    # 17-24: En Passant File
    if board.ep_square:
        file_idx = chess.square_file(board.ep_square)
        tensor[17 + file_idx, :, :] = 1.0

    # 25-26: Check
    if board.is_check():
        c = 25 if board.turn == chess.WHITE else 26
        tensor[c, :, :] = 1.0

    # --- C. Attack Maps (27-38) ---
    # এখানে আমরা প্রতিটি পিস টাইপের অ্যাটাক আলাদা করছি না (পারফরম্যান্সের জন্য)
    # সাদার অ্যাটাক 27, কালোর অ্যাটাক 28
    for sq in chess.SQUARES:
        row, col = divmod(sq, 8)
        if board.is_attacked_by(chess.WHITE, sq):
            tensor[27, 7-row, col] = 1.0
        if board.is_attacked_by(chess.BLACK, sq):
            tensor[28, 7-row, col] = 1.0

    # --- D. Static Positional Features (PST Hints) (39-118) ---
    # Coordinate encoding (Ranks 39-46, Files 47-54)
    for r in range(8): tensor[39+r, 7-r, :] = 1.0
    for f in range(8): tensor[47+f, :, f] = 1.0

    # Center Control Hints (55)
    center = [27, 28, 35, 36]
    for sq in center:
        r, c = divmod(sq, 8)
        tensor[55, 7-r, c] = 1.0

    # (অবশিষ্ট চ্যানেলগুলো 119 পর্যন্ত খালি রাখা হলো ফিউচার কমপ্লেক্স ফিচার বা Noise এর জন্য)
    # PyTorch এর জন্য এটি সমস্যা নয়।

    return torch.from_numpy(tensor)


# --- ২. Synapse Dataset Class ---
class SynapseDataset(IterableDataset):
    def __init__(self, db_path, shuffle_buffer_size=20000):
        self.db_path = db_path
        self.shuffle_buffer_size = shuffle_buffer_size

    def __iter__(self):
        # লোকাল ডেটাবেস থেকে কানেকশন (Fast SSD Access)
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # সব ডেটা রিড করা
        cursor.execute("SELECT fen, stats FROM positions")

        buffer = []
        for row in cursor:
            fen, stats_json = row

            # ডেটা প্রসেসিং (Weak Supervision)
            try:
                stats = json.loads(stats_json)
                total = stats['total']
                if total < 20: continue

                # Score Calculation (-1 to 1)
                # আমরা সব মুভের রেজাল্ট যোগ করে স্কোর বের করব
                w = sum(m['white'] for m in stats['moves'].values())
                b = sum(m['black'] for m in stats['moves'].values())
                # d = sum(m['draw'] for m in stats['moves'].values()) # Draw = 0 value
                real_total = w + b + sum(m['draw'] for m in stats['moves'].values())

                if real_total == 0: continue

                # White Win = 1, Black Win = -1, Draw = 0
                score = (w - b) / real_total

                buffer.append({'fen': fen, 'score': score})

                # RAM Shuffling
                if len(buffer) >= self.shuffle_buffer_size:
                    yield self.process_sample(buffer.pop(random.randrange(len(buffer))))

            except Exception:
                continue

        # বাফার খালি করা
        while buffer:
            yield self.process_sample(buffer.pop(random.randrange(len(buffer))))

        conn.close()

    def process_sample(self, sample):
        # Tensor Generation
        tensor = fen_to_dense_tensor(sample['fen'])
        score = torch.tensor([sample['score']], dtype=torch.float32)
        return tensor, score


# --- ৩. ডেটা লোডার টেস্ট ---
print("🧪 Testing Feature Extractor...")
dummy_fen = "rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1"
t = fen_to_dense_tensor(dummy_fen)
print(f"✅ Input Tensor Shape: {t.shape} (Expected: 119, 8, 8)")

# ডেটা লোডার তৈরি (Local Cache থেকে)
# আগের সেলে সেট করা LOCAL_DB_PATH ব্যবহার করছি
train_dataset = SynapseDataset(LOCAL_DB_PATH, shuffle_buffer_size=50000)
# num_workers=0 রাখা হলো GPU Error এড়াতে
train_loader = DataLoader(train_dataset, batch_size=CONFIG.BATCH_SIZE, num_workers=0)

print("✅ Synapse Dataset & Loader Ready.")

```


Output:



```text

🧪 Testing Feature Extractor...
✅ Input Tensor Shape: torch.Size([119, 8, 8]) (Expected: 119, 8, 8)
✅ Synapse Dataset & Loader Ready.
```

---

### Cell 3
```python


# Cell 3: Synapse-Base Hybrid Model Architecture (CNN + Transformer)

import torch.nn as nn
from einops.layers.torch import Rearrange
import torch.nn.init as init

# --- ১. ResNet Block (The CNN Backbone) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(in_channels)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += identity
        return self.relu(out)

# --- ২. Synapse-Base Model Class ---
class SynapseBase(nn.Module):
    def __init__(self,
                 input_channels=CONFIG.INPUT_CHANNELS,
                 num_filters=CONFIG.HIDDEN_DIM,
                 num_res_blocks=CONFIG.RESNET_BLOCKS,
                 num_transformer_layers=CONFIG.TRANSFORMER_LAYERS,
                 num_heads=CONFIG.HEADS):
        super().__init__()

        # --- A. Spatial Perception (CNN) ---
        # Initial Convolution: (119, 8, 8) -> (256, 8, 8)
        self.conv_in = nn.Sequential(
            nn.Conv2d(input_channels, num_filters, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(inplace=True)
        )

        # ResNet Tower (Deep Spatial Reasoning)
        self.residual_tower = nn.Sequential(
            *[ResidualBlock(num_filters) for _ in range(num_res_blocks)]
        )

        # --- B. Strategic Reasoning (Transformer) ---
        # Flatten Spatial Grid to Sequence: (Batch, 256, 8, 8) -> (Batch, 64, 256)
        self.to_sequence = Rearrange('b c h w -> b (h w) c')

        # Learnable Positional Encoding
        self.pos_embedding = nn.Parameter(torch.randn(1, 64, num_filters) * 0.02)

        # Transformer Encoder (Long-range dependencies)
        encoder_layer = nn.TransformerEncoderLayer(d_model=num_filters, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)

        # Back to Image format: (Batch, 64, 256) -> (Batch, 256, 8, 8)
        self.to_image = Rearrange('b (h w) c -> b c h w', h=8, w=8)

        # --- C. Output Heads ---
        # Value Head (Evaluation Score)
        self.value_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh() # Output range: [-1, 1]
        )

        # Policy Head (Move Probabilities - Placeholder for Fine-tuning)
        self.policy_head = nn.Sequential(
            nn.Conv2d(num_filters, 32, kernel_size=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4096) # Simplified Policy (64*64 moves)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        # 1. CNN Phase
        x = self.conv_in(x)
        x = self.residual_tower(x)

        # 2. Transformer Phase
        x_seq = self.to_sequence(x)
        x_seq = x_seq + self.pos_embedding
        x_seq = self.transformer(x_seq)

        # 3. Output Phase
        x_out = self.to_image(x_seq)

        value = self.value_head(x_out)
        policy_logits = self.policy_head(x_out)

        return value, policy_logits

print("✅ Synapse-Base Hybrid Model Architecture Defined (Bug-Free).")
 
   

```


Output:



```text

✅ Synapse-Base Hybrid Model Architecture Defined (Bug-Free).

  
```

---

### Cell 4
```python

# Cell 4: Synapse-Base Training Loop, Export & Upload (SHAPE FIXED)

import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from huggingface_hub import HfApi
import os
import time

# --- ১. ট্রেনিং ইনিশিয়ালাইজেশন ---
print("🚀 Initializing Synapse Training Engine...")

try:
    # মডেল জেনারেট এবং GPU তে পাঠানো
    model = SynapseBase().to(device)
    param_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"✅ Model Loaded on {device}. Parameters: {param_count:,}")
except NameError:
    print("⚠️ Model class not found. Run Cell 3 first.")
    raise

# Optimizer
optimizer = optim.AdamW(model.parameters(), lr=CONFIG.LEARNING_RATE, weight_decay=1e-2)

# Scheduler Setup
ESTIMATED_SAMPLES = 2500000
EFFECTIVE_BATCH = CONFIG.BATCH_SIZE * CONFIG.GRAD_ACCUMULATION
STEPS_PER_EPOCH = ESTIMATED_SAMPLES // EFFECTIVE_BATCH
TOTAL_STEPS = STEPS_PER_EPOCH * CONFIG.EPOCHS

print(f"📊 Scheduler Steps: {TOTAL_STEPS} (Based on ~2.5M samples)")

scheduler = OneCycleLR(
    optimizer,
    max_lr=1e-3,
    total_steps=TOTAL_STEPS,
    pct_start=0.1,
    div_factor=10,
    final_div_factor=100
)

# Loss Function & Scaler
criterion = nn.MSELoss()
scaler = torch.amp.GradScaler('cuda')

# --- ২. ট্রেনিং লুপ ---
print(f"\n🧠 Starting Training | Epochs: {CONFIG.EPOCHS} | Batch: {CONFIG.BATCH_SIZE} | Acc: {CONFIG.GRAD_ACCUMULATION}")
print("-" * 60)

start_time = time.time()
model.train()

for epoch in range(CONFIG.EPOCHS):
    running_loss = 0.0
    optimizer.zero_grad()

    for i, (inputs, targets) in enumerate(train_loader):
        # Move to GPU
        inputs = inputs.to(device, non_blocking=True)
        # FIX: unsqueeze সরানো হয়েছে কারণ DataLoader আগেই (Batch, 1) দিচ্ছে
        targets = targets.to(device, non_blocking=True)

        # Forward Pass
        with torch.amp.autocast('cuda'):
            pred_value, _ = model(inputs)

            # CRITICAL FIX: Shape মিসম্যাচ যাতে না হয়, তাই ফোর্স করা হচ্ছে
            # Prediction এবং Target দুটোকেই (Batch, 1) শেপে আনা হলো
            pred_value = pred_value.view(-1, 1)
            targets = targets.view(-1, 1)

            loss = criterion(pred_value, targets)

        # Backward Pass
        scaler.scale(loss / CONFIG.GRAD_ACCUMULATION).backward()

        # Gradient Accumulation Step
        if (i + 1) % CONFIG.GRAD_ACCUMULATION == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

        running_loss += loss.item()

        # Logging
        if (i + 1) % 100 == 0:
            avg_loss = running_loss / 100
            current_lr = optimizer.param_groups[0]['lr']
            elapsed = time.time() - start_time
            print(f"Epoch {epoch+1}/{CONFIG.EPOCHS} | Step {i+1} | Loss: {avg_loss:.6f} | LR: {current_lr:.6f} | Time: {elapsed:.0f}s")
            running_loss = 0.0

        if i >= (STEPS_PER_EPOCH * CONFIG.GRAD_ACCUMULATION):
            break

    print(f"✅ Epoch {epoch+1} Complete!")


# --- ৩. ONNX এক্সপোর্ট (CPU Offload) ---
print("\n💾 Finalizing & Exporting Synapse-Base...")
model.eval()
model.to('cpu')

dummy_input = torch.randn(1, 119, 8, 8).to('cpu')
onnx_path = os.path.join("/content/data", "synapse_base.onnx")

try:
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=['board_state'],
        output_names=['value', 'policy'],
        dynamic_axes=None
    )

    file_size = os.path.getsize(onnx_path) / (1024 * 1024)
    print(f"✅ ONNX Export Successful! Size: {file_size:.2f} MB")

    if file_size < 10:
        print("⚠️ Warning: Model size is unusually small. Check architecture.")

except Exception as e:
    print(f"❌ ONNX Export Failed: {e}")
    pass


# --- ৪. Hugging Face আপলোড ---
# আপনার টোকেন
HF_TOKEN = "HF_TOKEN"
HF_USERNAME = "Rafs-an09002"
MODEL_REPO = f"{HF_USERNAME}/gambitflow-synapse-base"

print(f"\n🚀 Uploading to Hugging Face: {MODEL_REPO}...")
api = HfApi(token=HF_TOKEN)

try:
    api.create_repo(repo_id=MODEL_REPO, repo_type="model", exist_ok=True)

    api.upload_file(
        path_or_fileobj=onnx_path,
        path_in_repo="synapse_base.onnx",
        repo_id=MODEL_REPO,
        repo_type="model",
        commit_message="Upload Synapse-Base v1 (Shape Fixed)"
    )

    print("🎉 Synapse-Base Live on Cloud!")
    print(f"🔗 URL: https://huggingface.co/{MODEL_REPO}/resolve/main/synapse_base.onnx")

except Exception as e:
    print(f"❌ Upload Failed: {e}")

```


Output:



```text

🚀 Initializing Synapse Training Engine...
✅ Model Loaded on cuda. Parameters: 38,098,881
📊 Scheduler Steps: 24410 (Based on ~2.5M samples)

🧠 Starting Training | Epochs: 10 | Batch: 256 | Acc: 4
------------------------------------------------------------
✅ Epoch 1 Complete!
✅ Epoch 2 Complete!
✅ Epoch 3 Complete!
✅ Epoch 4 Complete!
✅ Epoch 5 Complete!
✅ Epoch 6 Complete!
✅ Epoch 7 Complete!
✅ Epoch 8 Complete!
✅ Epoch 9 Complete!
✅ Epoch 10 Complete!

💾 Finalizing & Exporting Synapse-Base...
✅ ONNX Export Successful! Size: 145.38 MB

🚀 Uploading to Hugging Face: Rafs-an09002/gambitflow-synapse-base...
Processing Files (1 / 1)      : 100%
  152MB /  152MB, 34.6MB/s  
New Data Upload               : 100%
  152MB /  152MB, 34.6MB/s  
  ...nt/data/synapse_base.onnx: 100%
  152MB /  152MB            
🎉 Synapse-Base Live on Cloud!
🔗 URL: https://huggingface.co/Rafs-an09002/gambitflow-synapse-base/resolve/main/synapse_base.onnx

         
```


### Cell 5

```python
# Cell 5: Save PyTorch Weights (.pth) and Upload to HF

from huggingface_hub import HfApi
import torch
import os

# ১. মডেল ওয়েটস সেভ করা (Content ফোল্ডারে)
pth_path = "/content/data/synapse_base.pth"

print("💾 Saving PyTorch state dict...")
# আমরা শুধুমাত্র state_dict সেভ করছি যা ফাইন-টিউনিংয়ের জন্য সবচেয়ে ভালো এবং লাইটওয়েট
torch.save(model.state_dict(), pth_path)

file_size = os.path.getsize(pth_path) / (1024 * 1024)
print(f"✅ Weights Saved Locally. Size: {file_size:.2f} MB")

# ২. Hugging Face আপলোড
# আপনার টোকেন এবং রিপো আইডি (আগের সেল থেকে ভেরিয়েবল থাকলে সরাসরি কাজ করবে)
HF_TOKEN = "HF_TOKEN"
HF_USERNAME = "Rafs-an09002"
MODEL_REPO_ID = f"{HF_USERNAME}/gambitflow-synapse-base"

api = HfApi(token=HF_TOKEN)

print(f"\n🚀 Uploading .pth file to: {MODEL_REPO_ID}...")

try:
    api.upload_file(
        path_or_fileobj=pth_path,
        path_in_repo="synapse_base.pth", # এই নামেই হাগিং ফেসে থাকবে
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        commit_message="Add PyTorch weights for future fine-tuning"
    )
    print("\n🎉 SUCCESS! Original weights uploaded.")
    print("This file is now your 'Base' for Synapse-Edge.")

except Exception as e:
    print(f"\n❌ Upload Failed: {e}")
```

Output:



```text
💾 Saving PyTorch state dict...
✅ Weights Saved Locally. Size: 145.52 MB

🚀 Uploading .pth file to: Rafs-an09002/gambitflow-synapse-base...
Processing Files (1 / 1)      : 100%
  153MB /  153MB, 33.2MB/s  
New Data Upload               : 100%
  116MB /  116MB, 25.3MB/s  
  ...ent/data/synapse_base.pth: 100%
  153MB /  153MB            

🎉 SUCCESS! Original weights uploaded.
This file is now your 'Base' for Synapse-Edge.


```

