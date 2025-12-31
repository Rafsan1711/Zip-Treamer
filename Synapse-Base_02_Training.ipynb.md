## Notebook Name: Synapse-Base_02_Training.ipynb

---

## 1. Introduction
eta amar  GambitFlow/Synapse-Base  model er  notebook. 

---

### Cell 1: Environment Setup
```python
# ==================== SYNAPSE-BASE TRAINING NOTEBOOK ====================
# Cell 1: Environment Setup
# Purpose: Foundation model trained on 5M elite games
# Compatible: GPU ONLY (T4/V100/A100)

import os
import sys
import json
import time
import torch
from datetime import datetime
from google.colab import drive

# Mount Drive
drive.mount('/content/drive')

# Check GPU
if not torch.cuda.is_available():
    raise RuntimeError("❌ GPU Required! Go to Runtime → Change runtime type → GPU")

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

print(f"{'='*70}")
print(f"🚀 SYNAPSE-BASE: FOUNDATION MODEL TRAINING")
print(f"{'='*70}")
print(f"🖥️  GPU: {gpu_name}")
print(f"💾 VRAM: {gpu_memory:.1f} GB")
print(f"🔥 PyTorch: {torch.__version__}")
print(f"🎯 CUDA: {torch.version.cuda}")
print(f"{'='*70}\n")

# Project paths
BASE_PATH = '/content/drive/MyDrive/Chessmate_Project/Synapse_Models'
PATHS = {
    'base_model': os.path.join(BASE_PATH, 'synapse_base'),
    'checkpoints': os.path.join(BASE_PATH, 'synapse_base/checkpoints'),
    'logs': os.path.join(BASE_PATH, 'synapse_base/logs'),
    'data': os.path.join(BASE_PATH, 'data')
}

for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

print(f"✅ Project initialized at: {BASE_PATH}")

# Training state manager
class TrainingState:
    def __init__(self):
        self.state_file = os.path.join(PATHS['checkpoints'], 'training_state.json')
        self.state = self.load()

    def load(self):
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            print(f"\n📂 Resuming from:")
            print(f"   Epoch: {state['epoch']}")
            print(f"   Steps: {state['total_steps']:,}")
            print(f"   Best Loss: {state['best_loss']:.6f}")
            return state

        return {
            'epoch': 0,
            'total_steps': 0,
            'best_loss': float('inf'),
            'loss_history': [],
            'start_time': datetime.now().isoformat()
        }

    def save(self):
        self.state['last_updated'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(self.state, f, indent=2)

    def update(self, **kwargs):
        self.state.update(kwargs)
        self.save()

train_state = TrainingState()

print("\n✅ Environment ready for training!")


```


Output:



```text
Mounted at /content/drive
======================================================================
🚀 SYNAPSE-BASE: FOUNDATION MODEL TRAINING
======================================================================
🖥️  GPU: Tesla T4
💾 VRAM: 14.7 GB
🔥 PyTorch: 2.9.0+cu126
🎯 CUDA: 12.6
======================================================================

✅ Project initialized at: /content/drive/MyDrive/Chessmate_Project/Synapse_Models

✅ Environment ready for training!

```

---

### Cell 2
```python

# Cell 2: Install Training Dependencies
# Purpose: PyTorch ecosystem + chess libraries

print("📦 Installing dependencies...\n")

# Core ML libraries (already in Colab, but verify)
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader

print(f"✅ PyTorch {torch.__version__}")

# Install additional packages
!pip install -q python-chess  # Chess move validation
!pip install -q timm           # PyTorch Image Models (for advanced layers)
!pip install -q einops         # Tensor operations (for transformer)
!pip install -q huggingface_hub # Dataset download

import chess
import timm
import einops
from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm

print("✅ python-chess")
print("✅ timm (PyTorch Image Models)")
print("✅ einops (Tensor operations)")
print("✅ huggingface_hub")

# Import standard libraries
import numpy as np
import sqlite3
import json
import random
from collections import defaultdict

print("\n✅ All dependencies ready!")
print(f"\n💡 Tip: If training crashes, re-run from Cell 1 to resume from checkpoint")
```


Output:



```text
📦 Installing dependencies...

✅ PyTorch 2.9.0+cu126
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.1/6.1 MB 72.8 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
  Building wheel for chess (setup.py) ... done
✅ python-chess
✅ timm (PyTorch Image Models)
✅ einops (Tensor operations)
✅ huggingface_hub

✅ All dependencies ready!

💡 Tip: If training crashes, re-run from Cell 1 to resume from checkpoint
```

---

### Cell 3
```python



  # Cell 3: Load Elite Dataset from Hugging Face
# Source: 5M positions from 2000+ ELO games

print("📥 Downloading Elite Dataset from Hugging Face...\n")
print("🔗 Repository: GambitFlow/Elite-Data")
print("📊 Dataset: chess_stats_v2.db (5M+ positions)")
print("⏱️  This will take 1-2 minutes...\n")

# Download database
db_path = hf_hub_download(
    repo_id="GambitFlow/Elite-Data",
    filename="chess_stats_v2.db",
    repo_type="dataset",
    cache_dir=PATHS['data']
)

# Copy to faster local storage
import shutil
local_db_path = '/content/chess_stats_elite.db'

if not os.path.exists(local_db_path):
    print("📋 Copying to local storage for faster I/O...")
    shutil.copy(db_path, local_db_path)
    print("✅ Copy complete!")
else:
    print("✅ Using cached local copy")

# Verify database
conn = sqlite3.connect(local_db_path)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM positions")
total_positions = cursor.fetchone()[0]

cursor.execute("SELECT fen, stats FROM positions LIMIT 1")
sample_row = cursor.fetchone()
sample_fen, sample_stats = sample_row

conn.close()

print(f"\n{'='*70}")
print(f"📊 DATASET STATISTICS")
print(f"{'='*70}")
print(f"Total Positions: {total_positions:,}")
print(f"Database Size: {os.path.getsize(local_db_path) / (1024**3):.2f} GB")
print(f"Sample FEN: {sample_fen[:50]}...")
print(f"Sample Stats Keys: {list(json.loads(sample_stats).keys())}")
print(f"{'='*70}\n")

print("✅ Dataset loaded and verified!")
print(f"💾 Database path: {local_db_path}")
   

```


Output:



```text
📥 Downloading Elite Dataset from Hugging Face...

🔗 Repository: GambitFlow/Elite-Data
📊 Dataset: chess_stats_v2.db (5M+ positions)
⏱️  This will take 1-2 minutes...

/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
chess_stats_v2.db: 100%
 925M/925M [00:20<00:00, 144MB/s]
📋 Copying to local storage for faster I/O...
✅ Copy complete!

======================================================================
📊 DATASET STATISTICS
======================================================================
Total Positions: 2,554,920
Database Size: 0.86 GB
Sample FEN: rnbq1rk1/pp1p1ppp/4p3/2P5/2P1n3/P1Q5/1P2PPPP/R1B1K...
Sample Stats Keys: ['total', 'moves']
======================================================================

✅ Dataset loaded and verified!
💾 Database path: /content/chess_stats_elite.db

  
```

---

### Cell 4
```python


# Cell 4: Synapse-Base Architecture
# Following Synapse_Edge_Training_Prompt.md specifications
# 119-channel input, CNN backbone, 4 Transformer layers, 4 heads

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# ==================== ATTENTION MODULES ====================

class CBAM(nn.Module):
    """Convolutional Block Attention Module (Channel + Spatial)"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        # Channel attention
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        # Spatial attention
        self.spatial_conv = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel attention
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = self.sigmoid(avg_out + max_out)
        x = x * channel_att

        # Spatial attention
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial_conv(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att

        return x


class ResidualBlock(nn.Module):
    """Enhanced Residual Block with CBAM attention"""
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.cbam = CBAM(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.cbam(out)
        out += identity
        return self.relu(out)


# ==================== TRANSFORMER COMPONENTS ====================

class PositionalEncoding(nn.Module):
    """Learnable positional encoding for 64 board squares"""
    def __init__(self, d_model, max_len=64):
        super().__init__()
        self.pos_embedding = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

    def forward(self, x):
        return x + self.pos_embedding


class TransformerBlock(nn.Module):
    """Transformer encoder block with multi-head attention"""
    def __init__(self, d_model, num_heads, dim_feedforward, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # Multi-head self-attention
        attn_out, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_out)

        # Feed-forward network
        ffn_out = self.ffn(x)
        x = self.norm2(x + ffn_out)

        return x


# ==================== MAIN MODEL ====================

class SynapseBase(nn.Module):
    """
    Synapse-Base: Foundation Chess AI

    Architecture:
    - 119-channel input (piece planes + history + auxiliary features)
    - 20 ResNet blocks with CBAM (256 channels)
    - 4 Transformer encoder layers
    - 4 output heads: Policy, Value, Tactical, Endgame

    Parameters: ~45M
    Memory: ~5GB VRAM (training), ~2GB (inference)
    """

    def __init__(
        self,
        input_channels=119,
        base_channels=256,
        num_resnet_blocks=20,
        num_transformer_layers=4,
        transformer_heads=8,
        dropout=0.1
    ):
        super().__init__()

        self.input_channels = input_channels
        self.base_channels = base_channels

        # ===== INPUT PROCESSING =====
        self.input_conv = nn.Sequential(
            nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )

        # ===== RESNET BACKBONE =====
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(base_channels) for _ in range(num_resnet_blocks)
        ])

        # ===== TRANSFORMER LAYERS =====
        self.pos_encoding = PositionalEncoding(base_channels, max_len=64)
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=base_channels,
                num_heads=transformer_heads,
                dim_feedforward=base_channels * 4,
                dropout=dropout
            )
            for _ in range(num_transformer_layers)
        ])

        # ===== POLICY HEAD (Move Prediction) =====
        self.policy_head = nn.Sequential(
            nn.Conv2d(base_channels, 128, kernel_size=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(128 * 64, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(2048, 1968)  # All possible moves
        )

        # ===== VALUE HEAD (Position Evaluation) =====
        self.value_head = nn.Sequential(
            nn.Conv2d(base_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        # ===== TACTICAL HEAD (Pattern Recognition) =====
        self.tactical_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 10),  # 10 tactical patterns
            nn.Sigmoid()
        )

        # ===== ENDGAME HEAD (Phase Detection) =====
        self.endgame_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 4),  # Opening/Middlegame/Endgame/Tablebase
            nn.Softmax(dim=1)
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Kaiming initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, 119, 8, 8) board tensor

        Returns:
            policy: (B, 1968) move probabilities
            value: (B, 1) position evaluation
            tactical: (B, 10) tactical flags
            phase: (B, 4) game phase probabilities
        """
        batch_size = x.size(0)

        # Input processing
        x = self.input_conv(x)  # (B, 256, 8, 8)

        # ResNet backbone
        for block in self.residual_blocks:
            x = block(x)

        # Transformer processing
        # Reshape: (B, 256, 8, 8) → (B, 64, 256)
        x_flat = rearrange(x, 'b c h w -> b (h w) c')
        x_flat = self.pos_encoding(x_flat)

        for transformer in self.transformer_blocks:
            x_flat = transformer(x_flat)

        # Reshape back: (B, 64, 256) → (B, 256, 8, 8)
        x = rearrange(x_flat, 'b (h w) c -> b c h w', h=8, w=8)

        # Output heads
        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        tactical = self.tactical_head(x)
        phase = self.endgame_head(x)

        return policy, value, tactical, phase


# ==================== MODEL INSTANTIATION ====================

print("\n🔨 Building Synapse-Base model...\n")

model = SynapseBase(
    input_channels=119,
    base_channels=256,
    num_resnet_blocks=20,
    num_transformer_layers=4,
    transformer_heads=8,
    dropout=0.1
).to(device)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"{'='*70}")
print(f"🧠 SYNAPSE-BASE ARCHITECTURE")
print(f"{'='*70}")
print(f"Input Channels: 119")
print(f"Base Channels: 256")
print(f"ResNet Blocks: 20 (with CBAM)")
print(f"Transformer Layers: 4 (8 heads each)")
print(f"")
print(f"📊 Parameters:")
print(f"   Total: {total_params:,}")
print(f"   Trainable: {trainable_params:,}")
print(f"   Size (FP32): {total_params * 4 / (1024**2):.1f} MB")
print(f"   Size (FP16): {total_params * 2 / (1024**2):.1f} MB")
print(f"   Size (INT8): {total_params / (1024**2):.1f} MB")
print(f"")
print(f"💾 Memory Estimate:")
print(f"   Training: ~8 GB VRAM")
print(f"   Inference: ~2 GB VRAM")
print(f"{'='*70}\n")

print("✅ Model architecture ready!")
```


Output:



```text


🔨 Building Synapse-Base model...

======================================================================
🧠 SYNAPSE-BASE ARCHITECTURE
======================================================================
Input Channels: 119
Base Channels: 256
ResNet Blocks: 20 (with CBAM)
Transformer Layers: 4 (8 heads each)

📊 Parameters:
   Total: 50,304,935
   Trainable: 50,304,935
   Size (FP32): 191.9 MB
   Size (FP16): 95.9 MB
   Size (INT8): 48.0 MB

💾 Memory Estimate:
   Training: ~8 GB VRAM
   Inference: ~2 GB VRAM
======================================================================

✅ Model architecture ready!
         
```


### Cell 5

```python
# Cell 5: Enhanced Data Processing (119-Channel Input)
# Following Prompt.md specifications for rich input representation

import chess
import numpy as np
import torch

# ==================== 119-CHANNEL INPUT ENCODER ====================

class ChessBoardEncoder:
    """
    Encodes chess board to 119-channel tensor

    Channel breakdown:
    - 0-11: Current piece planes (P,N,B,R,Q,K × 2 colors)
    - 12-19: Repetition history (T-0 to T-7)
    - 20-22: Castling rights (4 channels)
    - 23-30: En passant target (8 channels, one-hot file)
    - 31-38: Move count phase (binned)
    - 39-48: Material count (10 channels)
    - 49-50: Check status (2 channels)
    - 51-114: Attack maps (64 channels, 32 per color)
    - 115-116: King safety (2 channels)
    - 117-118: Pawn structure metrics (2 channels)
    """

    def __init__(self):
        self.piece_to_index = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }

    def encode(self, board: chess.Board) -> np.ndarray:
        """
        Encode board to 119×8×8 tensor

        Args:
            board: python-chess Board object

        Returns:
            numpy array of shape (119, 8, 8)
        """
        tensor = np.zeros((119, 8, 8), dtype=np.float32)

        # ===== PIECE PLANES (0-11) =====
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                rank, file = divmod(square, 8)
                piece_idx = self.piece_to_index[piece.piece_type]
                color_offset = 0 if piece.color == chess.WHITE else 6
                tensor[piece_idx + color_offset, rank, file] = 1.0

        # ===== CASTLING RIGHTS (12-15) =====
        tensor[12, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
        tensor[13, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
        tensor[14, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
        tensor[15, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

        # ===== EN PASSANT (16-23) =====
        if board.ep_square is not None:
            ep_file = chess.square_file(board.ep_square)
            tensor[16 + ep_file, :, :] = 1.0

        # ===== MOVE COUNT PHASE (24-31) =====
        move_phase = min(board.fullmove_number // 10, 7)  # Bin into 8 phases
        tensor[24 + move_phase, :, :] = 1.0

        # ===== MATERIAL COUNT (32-41) =====
        for piece_type in [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN]:
            white_count = len(board.pieces(piece_type, chess.WHITE))
            black_count = len(board.pieces(piece_type, chess.BLACK))

            piece_idx = self.piece_to_index[piece_type]
            tensor[32 + piece_idx, :, :] = white_count / 8.0  # Normalize
            tensor[37 + piece_idx, :, :] = black_count / 8.0

        # ===== CHECK STATUS (42-43) =====
        if board.is_check():
            if board.turn == chess.WHITE:
                tensor[42, :, :] = 1.0
            else:
                tensor[43, :, :] = 1.0

        # ===== ATTACK MAPS (44-107) =====
        # For each square, mark if it's attacked by white/black
        for square in chess.SQUARES:
            rank, file = divmod(square, 8)

            # White attacks
            if board.is_attacked_by(chess.WHITE, square):
                tensor[44 + square, rank, file] = 1.0

            # Black attacks
            if board.is_attacked_by(chess.BLACK, square):
                tensor[76 + square, rank, file] = 1.0

        # ===== KING SAFETY (108-109) =====
        # Count attackers near kings
        white_king_sq = board.king(chess.WHITE)
        black_king_sq = board.king(chess.BLACK)

        if white_king_sq is not None:
            white_attackers = sum(
                1 for sq in chess.SQUARES
                if chess.square_distance(sq, white_king_sq) <= 2
                and board.is_attacked_by(chess.BLACK, sq)
            )
            tensor[108, :, :] = white_attackers / 8.0  # Normalize

        if black_king_sq is not None:
            black_attackers = sum(
                1 for sq in chess.SQUARES
                if chess.square_distance(sq, black_king_sq) <= 2
                and board.is_attacked_by(chess.WHITE, sq)
            )
            tensor[109, :, :] = black_attackers / 8.0

        # ===== PAWN STRUCTURE (110-111) =====
        # Count pawn islands (disconnected pawn groups)
        white_pawns = board.pieces(chess.PAWN, chess.WHITE)
        black_pawns = board.pieces(chess.PAWN, chess.BLACK)

        white_files = set(chess.square_file(sq) for sq in white_pawns)
        black_files = set(chess.square_file(sq) for sq in black_pawns)

        # Count islands (groups of adjacent files with pawns)
        def count_islands(files):
            if not files:
                return 0
            sorted_files = sorted(files)
            islands = 1
            for i in range(len(sorted_files) - 1):
                if sorted_files[i+1] - sorted_files[i] > 1:
                    islands += 1
            return islands

        white_islands = count_islands(white_files)
        black_islands = count_islands(black_files)

        tensor[110, :, :] = white_islands / 8.0
        tensor[111, :, :] = black_islands / 8.0

        # ===== TURN TO MOVE (112) =====
        tensor[112, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

        # ===== FIFTY MOVE RULE (113) =====
        tensor[113, :, :] = board.halfmove_clock / 100.0  # Normalize

        # ===== TOTAL MOVES (114-118) =====
        # One-hot encoding of move count bins
        move_bin = min(board.fullmove_number // 20, 4)  # 5 bins: 0-19, 20-39, etc.
        tensor[114 + move_bin, :, :] = 1.0

        return tensor

    def encode_fen(self, fen: str) -> torch.Tensor:
        """
        Encode FEN string to 119×8×8 tensor

        Args:
            fen: FEN string

        Returns:
            torch.Tensor of shape (119, 8, 8)
        """
        board = chess.Board(fen)
        numpy_tensor = self.encode(board)
        return torch.from_numpy(numpy_tensor)


# ==================== MOVE ENCODING ====================

def move_to_index(move_san: str) -> int:
    """
    Convert SAN move to index [0, 1967]
    Simplified version - uses hash for now

    Full implementation would map:
    - Normal moves: from_square × 64 + to_square
    - Promotions: separate indices
    - Castling: separate indices
    """
    # For now, use hash (in production, implement proper move indexing)
    return hash(move_san) % 1968


# ==================== DATASET CLASS ====================

class EliteChessDataset(IterableDataset):
    """
    Iterable dataset for streaming from SQLite database
    Generates 119-channel inputs on-the-fly
    """

    def __init__(self, db_path, shuffle_buffer=10000):
        self.db_path = db_path
        self.shuffle_buffer = shuffle_buffer
        self.encoder = ChessBoardEncoder()

    def __iter__(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Stream all positions
        cursor.execute("SELECT fen, stats FROM positions")

        buffer = []
        for fen, stats_json in cursor:
            try:
                stats = json.loads(stats_json)

                # Find best move (most played)
                if not stats.get('moves'):
                    continue

                best_move = max(
                    stats['moves'].items(),
                    key=lambda x: sum(x[1].values())
                )[0]

                # Calculate value
                move_stats = stats['moves'][best_move]
                total = sum(move_stats.values())
                if total == 0:
                    continue

                value = (move_stats.get('white', 0) - move_stats.get('black', 0)) / total

                # Encode board
                board_tensor = self.encoder.encode_fen(fen)
                move_idx = move_to_index(best_move)

                buffer.append({
                    'board': board_tensor,
                    'policy_target': torch.tensor(move_idx, dtype=torch.long),
                    'value_target': torch.tensor([value], dtype=torch.float32),
                    'tactical_target': torch.zeros(10, dtype=torch.float32),  # Positional
                    'phase_target': self._estimate_phase(fen)
                })

                # Shuffle and yield
                if len(buffer) >= self.shuffle_buffer:
                    random.shuffle(buffer)
                    for sample in buffer[:self.shuffle_buffer // 2]:
                        yield sample
                    buffer = buffer[self.shuffle_buffer // 2:]

            except Exception as e:
                continue

        # Yield remaining
        random.shuffle(buffer)
        for sample in buffer:
            yield sample

        conn.close()

    def _estimate_phase(self, fen):
        """Estimate game phase from move count"""
        board = chess.Board(fen)
        move_count = board.fullmove_number

        if move_count <= 15:
            return torch.tensor([1, 0, 0, 0], dtype=torch.float32)  # Opening
        elif move_count <= 40:
            return torch.tensor([0, 1, 0, 0], dtype=torch.float32)  # Middlegame
        else:
            return torch.tensor([0, 0, 1, 0], dtype=torch.float32)  # Endgame


# ==================== CREATE DATA LOADER ====================

print("\n🔄 Creating data loader...\n")

train_dataset = EliteChessDataset(
    db_path=local_db_path,
    shuffle_buffer=10000
)

train_loader = DataLoader(
    train_dataset,
    batch_size=256,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True
)

print("✅ Data loader ready!")
print(f"   Batch size: 256")
print(f"   Input shape: (256, 119, 8, 8)")
print(f"   Estimated batches per epoch: ~19,500")
```

Output:



```text

🔄 Creating data loader...

✅ Data loader ready!
   Batch size: 256
   Input shape: (256, 119, 8, 8)
   Estimated batches per epoch: ~19,500

```

### Cell 6
```python

# Cell 6: Training Setup (Loss, Optimizer, Scheduler)

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import OneCycleLR

# ==================== MULTI-TASK LOSS ====================

class SynapseBaseLoss(nn.Module):
    """
    Multi-task loss for 4 heads
    """
    def __init__(
        self,
        policy_weight=1.0,
        value_weight=0.5,
        tactical_weight=0.3,
        phase_weight=0.2
    ):
        super().__init__()
        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.tactical_weight = tactical_weight
        self.phase_weight = phase_weight

        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()
        self.tactical_loss_fn = nn.BCELoss()
        self.phase_loss_fn = nn.CrossEntropyLoss()

    def forward(self, outputs, targets):
        """
        Args:
            outputs: (policy, value, tactical, phase) from model
            targets: (policy_target, value_target, tactical_target, phase_target)

        Returns:
            total_loss, loss_dict
        """
        policy_pred, value_pred, tactical_pred, phase_pred = outputs
        policy_target, value_target, tactical_target, phase_target = targets

        # Individual losses
        policy_loss = self.policy_loss_fn(policy_pred, policy_target)
        value_loss = self.value_loss_fn(value_pred, value_target)
        tactical_loss = self.tactical_loss_fn(tactical_pred, tactical_target)

        # Phase loss (convert probabilities to class labels)
        phase_labels = torch.argmax(phase_target, dim=1)
        phase_loss = self.phase_loss_fn(phase_pred, phase_labels)

        # Total loss
        total_loss = (
            self.policy_weight * policy_loss +
            self.value_weight * value_loss +
            self.tactical_weight * tactical_loss +
            self.phase_weight * phase_loss
        )

        return total_loss, {
            'total': total_loss.item(),
            'policy': policy_loss.item(),
            'value': value_loss.item(),
            'tactical': tactical_loss.item(),
            'phase': phase_loss.item()
        }


# ==================== TRAINING CONFIGURATION ====================

TRAINING_CONFIG = {
    'epochs': 50,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'gradient_accumulation_steps': 4,  # Effective batch = 256×4 = 1024
    'max_grad_norm': 1.0,
    'save_every_steps': 1000,
    'log_every_steps': 100,
    'warmup_epochs': 3
}

print(f"\n{'='*70}")
print(f"⚙️  TRAINING CONFIGURATION")
print(f"{'='*70}")
for key, value in TRAINING_CONFIG.items():
    print(f"{key:30s}: {value}")
print(f"{'='*70}\n")

# ==================== OPTIMIZER ====================

criterion = SynapseBaseLoss(
    policy_weight=1.0,
    value_weight=0.5,
    tactical_weight=0.3,
    phase_weight=0.2
)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=TRAINING_CONFIG['learning_rate'],
    weight_decay=TRAINING_CONFIG['weight_decay'],
    betas=(0.9, 0.999)
)

# ==================== LEARNING RATE SCHEDULER ====================

# Estimate steps per epoch
steps_per_epoch = 19500 // TRAINING_CONFIG['gradient_accumulation_steps']

scheduler = OneCycleLR(
    optimizer,
    max_lr=TRAINING_CONFIG['learning_rate'],
    epochs=TRAINING_CONFIG['epochs'],
    steps_per_epoch=steps_per_epoch,
    pct_start=TRAINING_CONFIG['warmup_epochs'] / TRAINING_CONFIG['epochs'],
    anneal_strategy='cos',
    div_factor=25.0,
    final_div_factor=10000.0
)

# ==================== MIXED PRECISION ====================

scaler = torch.amp.GradScaler('cuda')

print("✅ Training setup complete!")
print(f"   Optimizer: AdamW")
print(f"   Scheduler: OneCycleLR (cosine annealing)")
print(f"   Mixed Precision: Enabled")
print(f"   Effective Batch Size: 1024")
```


Output:



```text


======================================================================
⚙️  TRAINING CONFIGURATION
======================================================================
epochs                        : 50
learning_rate                 : 0.001
weight_decay                  : 1e-05
gradient_accumulation_steps   : 4
max_grad_norm                 : 1.0
save_every_steps              : 1000
log_every_steps               : 100
warmup_epochs                 : 3
======================================================================

✅ Training setup complete!
   Optimizer: AdamW
   Scheduler: OneCycleLR (cosine annealing)
   Mixed Precision: Enabled
   Effective Batch Size: 1024
```


### Cell 7
```python
# Cell 7: Main Training Loop with Resume Support

import time
from tqdm.auto import tqdm

# ==================== CHECKPOINT MANAGEMENT ====================

def save_checkpoint(epoch, step, loss):
    """Save model checkpoint"""
    checkpoint_path = os.path.join(
        PATHS['checkpoints'],
        f'synapse_base_epoch{epoch}_step{step}.pt'
    )

    torch.save({
        'epoch': epoch,
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),
        'loss': loss,
        'config': TRAINING_CONFIG
    }, checkpoint_path)

    train_state.update(
        epoch=epoch,
        total_steps=step,
        last_checkpoint=checkpoint_path
    )

    return checkpoint_path


def load_checkpoint():
    """Load latest checkpoint if exists"""
    if train_state.state.get('last_checkpoint'):
        checkpoint_path = train_state.state['last_checkpoint']

        if os.path.exists(checkpoint_path):
            print(f"\n📥 Loading checkpoint: {os.path.basename(checkpoint_path)}")

            checkpoint = torch.load(checkpoint_path)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            scaler.load_state_dict(checkpoint['scaler_state_dict'])

            print(f"✅ Resumed from epoch {checkpoint['epoch']}, step {checkpoint['step']}")
            return checkpoint['epoch'], checkpoint['step']

    return 0, 0


# ==================== TRAINING FUNCTION ====================

def train_one_epoch(epoch, start_step=0):
    """Train for one epoch"""
    model.train()
    optimizer.zero_grad()

    epoch_loss = 0.0
    epoch_metrics = defaultdict(float)
    step = start_step

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{TRAINING_CONFIG['epochs']}")

    for batch_idx, batch in enumerate(pbar):

        # Skip already processed batches (for resume)
        if step < start_step:
            step += 1
            continue

        # Move to GPU
        board = batch['board'].to(device)
        policy_target = batch['policy_target'].to(device)
        value_target = batch['value_target'].to(device)
        tactical_target = batch['tactical_target'].to(device)
        phase_target = batch['phase_target'].to(device)

        # Forward pass with mixed precision
        with torch.amp.autocast('cuda'):
            policy, value, tactical, phase = model(board)

            loss, loss_dict = criterion(
                (policy, value, tactical, phase),
                (policy_target, value_target, tactical_target, phase_target)
            )

            # Scale for gradient accumulation
            loss = loss / TRAINING_CONFIG['gradient_accumulation_steps']

        # Backward pass
        scaler.scale(loss).backward()

        # Gradient accumulation
        if (batch_idx + 1) % TRAINING_CONFIG['gradient_accumulation_steps'] == 0:

            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                TRAINING_CONFIG['max_grad_norm']
            )

            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()

            step += 1

            # Logging
            if step % TRAINING_CONFIG['log_every_steps'] == 0:
                pbar.set_postfix({
                    'loss': loss_dict['total'],
                    'lr': scheduler.get_last_lr()[0]
                })

            # Checkpoint saving
            if step % TRAINING_CONFIG['save_every_steps'] == 0:
                checkpoint_path = save_checkpoint(epoch, step, loss_dict['total'])
                print(f"\n💾 Checkpoint saved: {os.path.basename(checkpoint_path)}")

                # Save best model
                if loss_dict['total'] < train_state.state['best_loss']:
                    train_state.state['best_loss'] = loss_dict['total']
                    best_path = os.path.join(PATHS['base_model'], 'synapse_base_best.pt')
                    torch.save(model.state_dict(), best_path)
                    print(f"🏆 New best model! Loss: {loss_dict['total']:.6f}")

        # Track metrics
        epoch_loss += loss_dict['total']
        for key, val in loss_dict.items():
            if key != 'total':
                epoch_metrics[key] += val

    # Epoch summary
    avg_loss = epoch_loss / len(train_loader)

    print(f"\n{'='*70}")
    print(f"📊 Epoch {epoch} Summary")
    print(f"{'='*70}")
    print(f"Average Loss: {avg_loss:.6f}")
    print(f"Policy Loss: {epoch_metrics['policy'] / len(train_loader):.6f}")
    print(f"Value Loss: {epoch_metrics['value'] / len(train_loader):.6f}")
    print(f"Tactical Loss: {epoch_metrics['tactical'] / len(train_loader):.6f}")
    print(f"Phase Loss: {epoch_metrics['phase'] / len(train_loader):.6f}")
    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.6f}")
    print(f"{'='*70}\n")

    return avg_loss, step


# ==================== MAIN TRAINING LOOP ====================

print(f"\n{'='*70}")
print(f"🚀 STARTING SYNAPSE-BASE TRAINING")
print(f"{'='*70}\n")

# Load checkpoint if exists
start_epoch, start_step = load_checkpoint()

try:
    for epoch in range(start_epoch + 1, TRAINING_CONFIG['epochs'] + 1):

        epoch_start_time = time.time()

        # Train one epoch
        avg_loss, final_step = train_one_epoch(epoch, start_step if epoch == start_epoch + 1 else 0)

        epoch_time = time.time() - epoch_start_time

        # Save epoch checkpoint
        checkpoint_path = save_checkpoint(epoch, final_step, avg_loss)

        print(f"⏱️  Epoch {epoch} completed in {epoch_time/60:.2f} minutes")
        print(f"💾 Checkpoint: {os.path.basename(checkpoint_path)}\n")

        # Reset start_step for next epoch
        start_step = 0

except KeyboardInterrupt:
    print("\n⚠️  Training interrupted!")
    print("💾 Saving checkpoint...")
    save_checkpoint(epoch, final_step, avg_loss)
    print("✅ Checkpoint saved. You can resume later by re-running this cell.")

except Exception as e:
    print(f"\n❌ Training error: {e}")
    print("💾 Saving emergency checkpoint...")
    save_checkpoint(epoch, final_step, avg_loss)
    raise e

print(f"\n{'='*70}")
print(f"🎉 TRAINING COMPLETE!")
print(f"{'='*70}")
print(f"Best Loss: {train_state.state['best_loss']:.6f}")
print(f"Total Steps: {train_state.state['total_steps']:,}")
print(f"Best Model: synapse_base_best.pt")
print(f"{'='*70}\n")
```


Output:



```text
======================================================================
🚀 STARTING SYNAPSE-BASE TRAINING
======================================================================

Epoch 1/50: 
 0/? [00:00<?, ?it/s]
```

