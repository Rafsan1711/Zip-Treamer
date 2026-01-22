## Notebook Name: official_training.ipynb.md
### cell 1
```python
# Cell 1: Environment Setup & Dependencies (FIXED)
# ==============================================================================
# 🧠 SYNAPSE-BASE TRAINING - INITIALIZATION
# ==============================================================================

import os
import sys
import time
import threading
import random
import numpy as np
from google.colab import drive

print("⚙️ Installing dependencies...")
# PyTorch (GPU version)
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 -q
# Other libraries (Removed 'sqlite3' as it's built-in)
!pip install python-chess huggingface_hub zstandard -q

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.amp import autocast, GradScaler # Updated import for newer PyTorch
import chess
import sqlite3
import json
from huggingface_hub import hf_hub_download

# Mount Drive for persistence
print("\n📁 Mounting Google Drive...")
drive.mount('/content/drive')

# Project paths
PROJECT_ROOT = '/content/drive/MyDrive/GambitFlow_Project'
CHECKPOINT_DIR = os.path.join(PROJECT_ROOT, 'Checkpoints')
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

print(f"✅ Checkpoint directory: {CHECKPOINT_DIR}")

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Device: {device}")

if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

# Keep-alive mechanism (prevents disconnection)
def keep_session_alive():
    """Background thread to keep Colab session active"""
    while True:
        time.sleep(60)
        # Dummy operation to keep session alive
        _ = 1 + 1

alive_thread = threading.Thread(target=keep_session_alive, daemon=True)
alive_thread.start()

print("✅ Keep-alive thread started")
print("✅ Setup complete. Ready for training!")
```
Output:

```text
⚙️ Installing dependencies...
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.1/6.1 MB 119.6 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
  Building wheel for chess (setup.py) ... done

📁 Mounting Google Drive...
Mounted at /content/drive
✅ Checkpoint directory: /content/drive/MyDrive/GambitFlow_Project/Checkpoints
🔧 Device: cuda
   GPU: Tesla T4
   Memory: 15.83 GB
✅ Keep-alive thread started
✅ Setup complete. Ready for training!

```


---


### cell 2
```python
# Cell 2: Download Training Databases from HuggingFace
# ==============================================================================
# 📥 DOWNLOAD ALL TRAINING DATA
# ==============================================================================

from huggingface_hub import hf_hub_download
import os
import shutil

# Local cache directory
CACHE_DIR = '/content/training_data'
os.makedirs(CACHE_DIR, exist_ok=True)

# Database mapping
DATABASES = {
    'opening': {
        'repo': 'GambitFlow/Opening-Database',
        'file': 'opening_theory.db',
        'location': 'huggingface'
    },
    'match': {
        'repo': 'GambitFlow/Elite-Data',
        'file': 'match_positions_v2.db',
        'location': 'huggingface'
    },
    'tactical': {
        'file': 'tactical_puzzles_v2.db',
        'location': 'drive'  # Puzzle DB is on Drive, not HuggingFace
    },
    'endgame': {
        'repo': 'GambitFlow/Endgame-Tablebase',
        'files': [
            'endgame/3_piece.db',
            'endgame/4_piece.db',
            'endgame/5_piece.db'
        ],
        'location': 'huggingface'
    }
}

db_paths = {}

print("📥 Downloading training databases from HuggingFace...")
print("=" * 60)

# Download opening database
try:
    print("\n1️⃣ Downloading Opening Database...")
    opening_path = hf_hub_download(
        repo_id=DATABASES['opening']['repo'],
        filename=DATABASES['opening']['file'],
        repo_type='dataset',
        cache_dir=CACHE_DIR
    )
    db_paths['opening'] = opening_path
    size_mb = os.path.getsize(opening_path) / (1024**2)
    print(f"   ✅ Opening DB: {size_mb:.2f} MB")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    db_paths['opening'] = None

# Download match database
try:
    print("\n2️⃣ Downloading Match Position Database...")
    match_path = hf_hub_download(
        repo_id=DATABASES['match']['repo'],
        filename=DATABASES['match']['file'],
        repo_type='dataset',
        cache_dir=CACHE_DIR
    )
    db_paths['match'] = match_path
    size_mb = os.path.getsize(match_path) / (1024**2)
    print(f"   ✅ Match DB: {size_mb:.2f} MB")
except Exception as e:
    print(f"   ❌ Failed: {e}")
    db_paths['match'] = None

# Download tactical database (from Drive)
try:
    print("\n3️⃣ Loading Tactical Puzzle Database from Drive...")
    tactical_filename = DATABASES['tactical']['file']

    # Check in Synapse_Data directory
    tactical_path = os.path.join(PROJECT_ROOT, 'Synapse_Data', tactical_filename)

    if os.path.exists(tactical_path):
        db_paths['tactical'] = tactical_path
        size_mb = os.path.getsize(tactical_path) / (1024**2)
        print(f"   ✅ Tactical DB: {size_mb:.2f} MB")
        print(f"   📁 Path: {tactical_path}")
    else:
        print(f"   ❌ File not found at: {tactical_path}")
        print(f"   ⚠️  Please ensure {tactical_filename} is in {PROJECT_ROOT}/Synapse_Data/")
        db_paths['tactical'] = None
except Exception as e:
    print(f"   ❌ Failed: {e}")
    db_paths['tactical'] = None

# Download endgame databases
print("\n4️⃣ Downloading Endgame Databases...")
endgame_paths = []
for file in DATABASES['endgame']['files']:
    try:
        endgame_path = hf_hub_download(
            repo_id=DATABASES['endgame']['repo'],
            filename=file,
            repo_type='dataset',
            cache_dir=CACHE_DIR
        )
        endgame_paths.append(endgame_path)
        size_mb = os.path.getsize(endgame_path) / (1024**2)
        print(f"   ✅ {file}: {size_mb:.2f} MB")
    except Exception as e:
        print(f"   ❌ Failed {file}: {e}")

db_paths['endgame'] = endgame_paths if endgame_paths else None

# Verification
print("\n" + "=" * 60)
print("📊 Database Status:")
for name, path in db_paths.items():
    if path:
        if isinstance(path, list):
            print(f"   ✅ {name.upper()}: {len(path)} files loaded")
        else:
            print(f"   ✅ {name.upper()}: Ready")
    else:
        print(f"   ❌ {name.upper()}: Missing (training will skip this source)")

# Save paths for later use
import pickle
paths_file = os.path.join(CHECKPOINT_DIR, 'db_paths.pkl')
with open(paths_file, 'wb') as f:
    pickle.dump(db_paths, f)

print(f"\n✅ Database paths saved to: {paths_file}")
print("✅ Data download complete!")
print("\n⚠️  If any database failed, check your HuggingFace repository names")
```

Output:

```text
 📥 Downloading training databases from HuggingFace...
============================================================

1️⃣ Downloading Opening Database...
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
opening_theory.db: 100% 2.75M/2.75M [00:01<00:00, 1.78MB/s]   ✅ Opening DB: 2.62 MB

2️⃣ Downloading Match Position Database...
match_positions_v2.db: 100% 1.23G/1.23G [00:13<00:00, 168MB/s]   ✅ Match DB: 1177.25 MB

3️⃣ Loading Tactical Puzzle Database from Drive...
   ✅ Tactical DB: 194.28 MB
   📁 Path: /content/drive/MyDrive/GambitFlow_Project/Synapse_Data/tactical_puzzles_v2.db

4️⃣ Downloading Endgame Databases...
endgame/3_piece.db: 100% 28.4M/28.4M [00:00<00:00, 44.7MB/s]   ✅ endgame/3_piece.db: 27.09 MB
endgame/4_piece.db: 100% 16.1M/16.1M [00:00<00:00, 25.7MB/s]   ✅ endgame/4_piece.db: 15.36 MB
endgame/5_piece.db: 100% 32.6M/32.6M [00:00<00:00, 42.7MB/s]   ✅ endgame/5_piece.db: 31.14 MB

============================================================
📊 Database Status:
   ✅ OPENING: Ready
   ✅ MATCH: Ready
   ✅ TACTICAL: Ready
   ✅ ENDGAME: 3 files loaded

✅ Database paths saved to: /content/drive/MyDrive/GambitFlow_Project/Checkpoints/db_paths.pkl
✅ Data download complete!

⚠️  If any database failed, check your HuggingFace repository names

```
---

### cell 3
```python

# Cell 3: Board Encoding Utilities (28 Channels)
# ==============================================================================
# 🎨 BOARD REPRESENTATION FOR NEURAL NETWORK
# ==============================================================================

import chess
import numpy as np
import torch

def fen_to_28_channels(fen):
    """
    Convert chess position (FEN) to 28-channel tensor representation.

    Channel Layout:
    - Channels 0-5: White pieces (P, N, B, R, Q, K)
    - Channels 6-11: Black pieces (p, n, b, r, q, k)
    - Channel 12: Turn (1.0 if white's turn, 0.0 if black's)
    - Channels 13-16: Castling rights (KQkq)
    - Channels 17-19: En passant file (one-hot encoding for files a-h, grouped)
    - Channels 20-27: Reserved for move history (not implemented yet)

    Args:
        fen (str): Position in FEN notation

    Returns:
        numpy.ndarray: Shape (28, 8, 8), dtype float32
    """
    board = chess.Board(fen)
    tensor = np.zeros((28, 8, 8), dtype=np.float32)

    # Piece type to channel mapping
    piece_map = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5
    }

    # Channels 0-11: Piece positions
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece:
            piece_type = piece.piece_type
            channel = piece_map[piece_type]

            # Black pieces offset by 6 channels
            if piece.color == chess.BLACK:
                channel += 6

            # Convert square to rank/file (chess uses rank 0 = bottom)
            # We flip to make it consistent (rank 0 = top)
            rank = 7 - chess.square_rank(square)
            file = chess.square_file(square)

            tensor[channel, rank, file] = 1.0

    # Channel 12: Turn indicator
    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0
    else:
        tensor[12, :, :] = 0.0

    # Channels 13-16: Castling rights
    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, :, :] = 1.0

    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[14, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[15, :, :] = 1.0

    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[16, :, :] = 1.0

    # Channels 17-19: En passant file (simplified 3-channel encoding)
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        # Group files into 3 channels: a-b-c (0), d-e-f (1), g-h (2)
        channel_idx = 17 + min(ep_file // 3, 2)
        tensor[channel_idx, :, :] = 1.0

    # Channels 20-27: Move history (reserved for future implementation)
    # Currently filled with zeros

    return tensor


def move_to_index(move_uci, board=None):
    """
    Convert UCI move to index for policy head.

    Policy head has 4096 outputs (64 from_squares × 64 to_squares).
    This is simplified - doesn't include promotions separately.

    Args:
        move_uci (str): Move in UCI format (e.g., 'e2e4')
        board (chess.Board, optional): Board object if available

    Returns:
        int: Index in range [0, 4095]
    """
    if isinstance(move_uci, chess.Move):
        move = move_uci
    else:
        move = chess.Move.from_uci(move_uci)

    from_sq = move.from_square
    to_sq = move.to_square

    # Simple encoding: from_square * 64 + to_square
    index = from_sq * 64 + to_sq

    return index


def index_to_move(index):
    """
    Convert policy index back to move.

    Args:
        index (int): Index in range [0, 4095]

    Returns:
        chess.Move: Corresponding move
    """
    from_sq = index // 64
    to_sq = index % 64

    return chess.Move(from_sq, to_sq)


# Test the encoding
print("🧪 Testing board encoding...")

# Test position: Starting position
test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
tensor = fen_to_28_channels(test_fen)

print(f"✅ Tensor shape: {tensor.shape}")
print(f"✅ Tensor dtype: {tensor.dtype}")
print(f"✅ Memory usage: {tensor.nbytes / 1024:.2f} KB")

# Verify piece positions
board = chess.Board(test_fen)
print("\n🔍 Verification:")
print(f"   White queen should be on d1 (channel 4, rank 7, file 3)")
print(f"   Value at [4, 7, 3]: {tensor[4, 7, 3]}")

print(f"   Black queen should be on d8 (channel 10, rank 0, file 3)")
print(f"   Value at [10, 0, 3]: {tensor[10, 0, 3]}")

print(f"   Turn channel (should be 1.0 for white)")
print(f"   Value at [12, 0, 0]: {tensor[12, 0, 0]}")

# Test move encoding
test_move = chess.Move.from_uci("e2e4")
move_idx = move_to_index(test_move)
reconstructed = index_to_move(move_idx)

print(f"\n   Test move: {test_move.uci()}")
print(f"   Encoded index: {move_idx}")
print(f"   Reconstructed: {reconstructed.uci()}")
print(f"   Match: {test_move == reconstructed}")

print("\n✅ Board encoding utilities ready!")
print("⚠️  Channels 20-27 (move history) are reserved for future use")


```

Output:

```text
🧪 Testing board encoding...
✅ Tensor shape: (28, 8, 8)
✅ Tensor dtype: float32
✅ Memory usage: 7.00 KB

🔍 Verification:
   White queen should be on d1 (channel 4, rank 7, file 3)
   Value at [4, 7, 3]: 1.0
   Black queen should be on d8 (channel 10, rank 0, file 3)
   Value at [10, 0, 3]: 1.0
   Turn channel (should be 1.0 for white)
   Value at [12, 0, 0]: 1.0

   Test move: e2e4
   Encoded index: 796
   Reconstructed: e2e4
   Match: True

✅ Board encoding utilities ready!
⚠️  Channels 20-27 (move history) are reserved for future use

```
---


### cell 4
```python


# Cell 4: Synapse-Base Model Architecture
# ==============================================================================
# 🧠 NEURAL NETWORK DEFINITION
# ==============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    """
    Residual block with skip connection.
    Architecture: Conv -> BN -> ReLU -> Conv -> BN -> Add -> ReLU
    """
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()

        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)

        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity  # Skip connection
        out = F.relu(out)

        return out


class SynapseBase(nn.Module):
    """
    Synapse-Base: Lightweight chess engine neural network.

    Architecture:
    - Input: (28, 8, 8) - 28 channels of board representation
    - Conv stem: 28 -> 128 filters
    - Residual tower: 15 ResBlocks with 128 filters
    - Value head: Evaluates position (-1 to +1)
    - Opening policy head: Predicts best opening moves (1968 common moves)
    - Tactical policy head: Full move prediction (4096 possible moves)

    Target size: 20-30 MB
    Target inference: <100ms on CPU
    """

    def __init__(self, num_res_blocks=15, num_filters=128):
        super(SynapseBase, self).__init__()

        # === FEATURE EXTRACTION ===
        # Convolutional stem
        self.conv_stem = nn.Conv2d(28, num_filters, kernel_size=3, padding=1, bias=False)
        self.bn_stem = nn.BatchNorm2d(num_filters)

        # Residual tower
        self.res_blocks = nn.ModuleList([
            ResidualBlock(num_filters) for _ in range(num_res_blocks)
        ])

        # === VALUE HEAD ===
        # Predicts position evaluation from current player's perspective
        self.value_conv = nn.Conv2d(num_filters, 32, kernel_size=1, bias=False)
        self.value_bn = nn.BatchNorm2d(32)
        self.value_fc1 = nn.Linear(32 * 8 * 8, 256)
        self.value_fc2 = nn.Linear(256, 1)

        # === OPENING POLICY HEAD ===
        # For opening phase (common moves only)
        self.opening_conv = nn.Conv2d(num_filters, 64, kernel_size=1, bias=False)
        self.opening_bn = nn.BatchNorm2d(64)
        self.opening_fc = nn.Linear(64 * 8 * 8, 1968)  # Common opening moves

        # === TACTICAL POLICY HEAD ===
        # For tactical/middlegame/endgame (all possible moves)
        self.tactical_conv = nn.Conv2d(num_filters, 64, kernel_size=1, bias=False)
        self.tactical_bn = nn.BatchNorm2d(64)
        self.tactical_fc = nn.Linear(64 * 8 * 8, 4096)  # 64x64 from-to squares

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Tensor of shape (batch_size, 28, 8, 8)

        Returns:
            value: Tensor of shape (batch_size, 1) in range [-1, 1]
            opening_policy: Tensor of shape (batch_size, 1968)
            tactical_policy: Tensor of shape (batch_size, 4096)
        """
        # Feature extraction
        x = self.conv_stem(x)
        x = self.bn_stem(x)
        x = F.relu(x)

        # Residual tower
        for block in self.res_blocks:
            x = block(x)

        # === VALUE HEAD ===
        value = self.value_conv(x)
        value = self.value_bn(value)
        value = F.relu(value)
        value = value.view(value.size(0), -1)  # Flatten
        value = F.relu(self.value_fc1(value))
        value = torch.tanh(self.value_fc2(value))  # Output in [-1, 1]

        # === OPENING POLICY HEAD ===
        opening = self.opening_conv(x)
        opening = self.opening_bn(opening)
        opening = F.relu(opening)
        opening = opening.view(opening.size(0), -1)  # Flatten
        opening_logits = self.opening_fc(opening)  # Raw logits

        # === TACTICAL POLICY HEAD ===
        tactical = self.tactical_conv(x)
        tactical = self.tactical_bn(tactical)
        tactical = F.relu(tactical)
        tactical = tactical.view(tactical.size(0), -1)  # Flatten
        tactical_logits = self.tactical_fc(tactical)  # Raw logits

        return value, opening_logits, tactical_logits


# === MODEL TESTING ===
print("🏗️  Initializing Synapse-Base architecture...")

model = SynapseBase(num_res_blocks=15, num_filters=128)

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"✅ Model created successfully!")
print(f"\n📊 Model Statistics:")
print(f"   Total parameters: {total_params:,}")
print(f"   Trainable parameters: {trainable_params:,}")
print(f"   Estimated size: {total_params * 4 / (1024**2):.2f} MB (FP32)")
print(f"   Estimated size: {total_params * 2 / (1024**2):.2f} MB (FP16)")

# Test forward pass
print("\n🧪 Testing forward pass...")
dummy_input = torch.randn(2, 28, 8, 8)  # Batch of 2

with torch.no_grad():
    value, opening_logits, tactical_logits = model(dummy_input)

print(f"✅ Forward pass successful!")
print(f"   Value shape: {value.shape} (expected: [2, 1])")
print(f"   Opening policy shape: {opening_logits.shape} (expected: [2, 1968])")
print(f"   Tactical policy shape: {tactical_logits.shape} (expected: [2, 4096])")
print(f"   Value range: [{value.min().item():.3f}, {value.max().item():.3f}]")

# Move to GPU if available
if torch.cuda.is_available():
    model = model.to(device)
    print(f"\n✅ Model moved to {device}")

    # Test GPU forward pass
    dummy_gpu = dummy_input.to(device)
    with torch.no_grad():
        _ = model(dummy_gpu)
    print("✅ GPU inference working!")

print("\n✅ Architecture ready for training!")
print(f"🎯 Target: {trainable_params / 1e6:.1f}M parameters, ~25 MB ONNX")

```

Output:

```text

🏗️  Initializing Synapse-Base architecture...
✅ Model created successfully!

📊 Model Statistics:
   Total parameters: 29,853,681
   Trainable parameters: 29,853,681
   Estimated size: 113.88 MB (FP32)
   Estimated size: 56.94 MB (FP16)

🧪 Testing forward pass...
✅ Forward pass successful!
   Value shape: torch.Size([2, 1]) (expected: [2, 1])
   Opening policy shape: torch.Size([2, 1968]) (expected: [2, 1968])
   Tactical policy shape: torch.Size([2, 4096]) (expected: [2, 4096])
   Value range: [0.089, 0.147]

✅ Model moved to cuda
✅ GPU inference working!

✅ Architecture ready for training!
🎯 Target: 29.9M parameters, ~25 MB ONNX




```
---

### cell 5
```python

# ==============================================================================
# 📊 ULTIMATE DATASET LOADER (V4 - PRODUCTION READY)
# ==============================================================================

import sqlite3
import json
import random
import numpy as np
import torch
from torch.utils.data import IterableDataset
import chess
import os
import pickle
import time

# --- 1. Load Database Paths ---
paths_file = os.path.join(CHECKPOINT_DIR, 'db_paths.pkl')
with open(paths_file, 'rb') as f:
    db_paths = pickle.load(f)

print("📂 Loaded database paths.")

# --- 2. Configuration ---
SAMPLING_WEIGHTS = {
    'opening': 0.15, 'match': 0.55, 'tactical': 0.20, 'endgame': 0.10
}
MATCH_DB_CAP = 3_000_000 # Use a maximum of 3M positions from the match DB

# --- 3. The Dataset Class ---
class SynapseDataset(IterableDataset):
    """
    A production-ready, highly robust IterableDataset for PyTorch.
    - Handles multiple data sources with weighted sampling.
    - Safe for multiprocessing (num_workers > 0 in DataLoader).
    - Efficiently samples from large databases.
    - Gracefully handles data parsing errors.
    """
    def __init__(self, db_paths, weights, match_cap):
        super(SynapseDataset).__init__()
        self.db_paths = db_paths
        self.weights = weights
        self.match_cap = match_cap

        # Connections and sizes are initialized per-worker
        self.connections = {}
        self.db_sizes = {}

    def _initialize_worker(self):
        """
        Executed by each DataLoader worker to get its own DB connections and info.
        This is the key to safe multiprocessing with SQLite.
        """
        worker_info = torch.utils.data.get_worker_info()
        seed = worker_info.seed if worker_info else int(time.time())
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))

        for source, path_or_paths in self.db_paths.items():
            if not path_or_paths: continue

            try:
                if isinstance(path_or_paths, list):
                    self.connections[source] = [sqlite3.connect(f'file:{p}?mode=ro', uri=True) for p in path_or_paths]
                    self.db_sizes[source] = sum(self._get_count(conn) for conn in self.connections[source])
                else:
                    conn = sqlite3.connect(f'file:{path_or_paths}?mode=ro', uri=True)
                    self.connections[source] = conn
                    table = 'puzzles' if source == 'tactical' else 'openings' if source == 'opening' else 'positions'
                    self.db_sizes[source] = self._get_count(conn, table)
            except Exception as e:
                print(f"Worker {worker_info.id if worker_info else 0} failed to connect to {source}: {e}")

        # Adjust weights based on successfully loaded DBs
        valid_weights = {k: v for k, v in self.weights.items() if k in self.db_sizes and self.db_sizes[k] > 0}
        total_weight = sum(valid_weights.values())
        self.sources = list(valid_weights.keys())
        self.source_weights = [w / total_weight for w in valid_weights.values()]

    def _get_count(self, conn, table_name='positions'):
        """Gets row count from a table."""
        try:
            return conn.cursor().execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        except:
            # Try other common table names
            for name in ['openings', 'puzzles', 'endgame']:
                try: return conn.cursor().execute(f"SELECT COUNT(*) FROM {name}").fetchone()[0]
                except: continue
        return 0

    # --- Sampler Functions ---
    def _sample(self, source, query_template):
        conns = self.connections.get(source)
        if not conns: return None
        conn = random.choice(conns) if isinstance(conns, list) else conns

        # Use random offset for large tables
        total_rows = self.db_sizes[source] if source != 'endgame' else self._get_count(conn, 'endgame')
        if total_rows == 0: return None
        offset = random.randint(0, total_rows - 1)

        # For match DB, respect the cap
        if source == 'match':
            offset = random.randint(0, min(total_rows, self.match_cap) - 1)

        return conn.cursor().execute(query_template, (offset,)).fetchone()

    def _process_sample(self, source):
        try:
            if source == 'opening':
                row = self._sample(source, "SELECT fen, move_data FROM openings LIMIT 1 OFFSET ?")
                if not row: return None
                fen, move_data = row[0], json.loads(row[1])
                moves, freqs = list(move_data.keys()), [d.get('frequency', 1) for d in move_data.values()]
                if not moves: return None
                selected_move = random.choices(moves, weights=freqs, k=1)[0]
                score = move_data[selected_move].get('avg_score', 0.5)
                move_idx = move_to_index(chess.Board(fen).parse_san(selected_move))
                return fen_to_28_channels(fen), (score - 0.5) * 2.0, move_idx

            elif source == 'match':
                row = self._sample(source, "SELECT fen, value_target, move_played FROM positions LIMIT 1 OFFSET ?")
                if not row: return None
                fen, value, move_san = row
                move_idx = move_to_index(chess.Board(fen).parse_san(move_san))
                return fen_to_28_channels(fen), value, move_idx

            elif source == 'tactical':
                row = self._sample(source, "SELECT fen, moves FROM puzzles LIMIT 1 OFFSET ?")
                if not row: return None
                fen, moves_uci = row
                move_idx = move_to_index(chess.Move.from_uci(moves_uci.split()[0]))
                return fen_to_28_channels(fen), 1.0, move_idx

            elif source == 'endgame':
                # Endgame tables might be smaller, ORDER BY RANDOM() is fine here.
                conns = self.connections.get('endgame')
                if not conns: return None
                conn = random.choice(conns)
                row = conn.cursor().execute("SELECT fen, wdl FROM endgame ORDER BY RANDOM() LIMIT 1").fetchone()
                if not row: return None
                fen, wdl = row
                return fen_to_28_channels(fen), float(wdl), 0 # Dummy move index

        except (ValueError, KeyError, IndexError, json.JSONDecodeError, chess.InvalidMoveError, chess.IllegalMoveError):
            return None # Gracefully handle any data parsing error

    def __iter__(self):
        self._initialize_worker()

        while True:
            # Select a data source based on weights
            source = random.choices(self.sources, weights=self.source_weights, k=1)[0]

            # Attempt to get a valid sample from the chosen source
            sample_data = self._process_sample(source)

            if sample_data:
                features, value, move_idx = sample_data
                yield (torch.from_numpy(features),
                       torch.tensor([value], dtype=torch.float32),
                       torch.tensor(move_idx, dtype=torch.long),
                       source)

# --- 4. Final Verification ---
print("\n🧪 Verifying the Production-Ready Dataset Loader...")

# Create a test instance of the dataset
test_dataset = SynapseDataset(db_paths, SAMPLING_WEIGHTS, MATCH_DB_CAP)

# Simulate sampling to check distribution and for errors
source_counts = {s: 0 for s in SAMPLING_WEIGHTS.keys()}
num_samples_to_test = 1000
iterator = iter(test_dataset)

print(f"\n📊 Sampling {num_samples_to_test} examples to verify distribution (this might take a moment)...")
for i in range(num_samples_to_test):
    try:
        features, value, move_idx, source = next(iterator)
        source_counts[source] += 1
        if i < 5:
             print(f"   Sample {i+1}: Source={source:<10} | Value={value.item():+.2f} | Move Index={move_idx.item()}")
    except StopIteration:
        print("   [Warning] Iterator stopped unexpectedly.")
        break

print("\n📊 Distribution Check:")
for source in SAMPLING_WEIGHTS.keys():
    if source in test_dataset.source_weights:
        actual_pct = 100 * source_counts.get(source, 0) / num_samples_to_test
        expected_pct = [w for s, w in zip(test_dataset.sources, test_dataset.source_weights) if s == source][0] * 100
        status = "✅" if abs(actual_pct - expected_pct) < 5 else "⚠️"
        print(f"   {status} {source.capitalize():<10}: {actual_pct:.1f}% (Expected: ~{expected_pct:.1f}%)")

print("\n✅ Dataset logic is production-ready.")


```

Output:

```text

📂 Loaded database paths.

🧪 Verifying the Production-Ready Dataset Loader...

📊 Sampling 1000 examples to verify distribution (this might take a moment)...
   Sample 1: Source=endgame    | Value=-1.00 | Move Index=0
   Sample 2: Source=endgame    | Value=+1.00 | Move Index=0
   Sample 3: Source=endgame    | Value=-1.00 | Move Index=0
   Sample 4: Source=opening    | Value=-0.20 | Move Index=3112
   Sample 5: Source=match      | Value=+1.00 | Move Index=2682

📊 Distribution Check:

✅ Dataset logic is production-ready.




```
---

### cell 6
```python


# Cell 6: Training Configuration & Checkpoint Management
# ==============================================================================
# ⚙️ TRAINING SETUP WITH AUTO-RESUME CAPABILITY
# ==============================================================================

import os
import json
import torch
from datetime import datetime

# === TRAINING HYPERPARAMETERS ===
CONFIG = {
    # Model
    'num_res_blocks': 15,
    'num_filters': 128,

    # Training
    'batch_size': 512,
    'accumulation_steps': 2,  # Effective batch = 512 * 2 = 1024
    'epochs': 10,
    'total_steps': 200000,  # Stop after this many steps regardless of epochs

    # Optimizer
    'learning_rate': 1e-4,
    'weight_decay': 1e-4,
    'betas': (0.9, 0.999),
    'grad_clip': 1.0,

    # Learning rate schedule
    'warmup_steps': 1000,
    'lr_schedule': 'onecycle',  # 'onecycle' or 'cosine' or 'constant'

    # Loss weights
    'loss_value_weight': 1.0,
    'loss_policy_weight': 0.5,

    # Checkpointing
    'save_every_steps': 2000,  # Save every N steps
    'validate_every_steps': 5000,  # Run validation every N steps
    'keep_last_n_checkpoints': 5,

    # Mixed precision
    'use_amp': True,  # Automatic Mixed Precision (FP16)

    # Dataset
    'num_workers': 2,
    'prefetch_factor': 2,
}

print("⚙️  Training Configuration:")
print("=" * 60)
for key, value in CONFIG.items():
    print(f"   {key:30s}: {value}")
print("=" * 60)


class CheckpointManager:
    """
    Manages training checkpoints with auto-resume capability.

    Saves to Google Drive for persistence across Colab disconnections.
    """

    def __init__(self, checkpoint_dir, config):
        self.checkpoint_dir = checkpoint_dir
        self.config = config
        self.checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint.json')

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save_checkpoint(self, model, optimizer, scheduler, scaler, step, epoch, best_loss):
        """Save training checkpoint"""

        checkpoint = {
            'step': step,
            'epoch': epoch,
            'best_loss': best_loss,
            'config': self.config,
            'timestamp': datetime.now().isoformat()
        }

        # Save model state
        model_path = os.path.join(self.checkpoint_dir, f'model_step_{step}.pth')
        torch.save(model.state_dict(), model_path)

        # Save optimizer state
        optimizer_path = os.path.join(self.checkpoint_dir, f'optimizer_step_{step}.pth')
        torch.save(optimizer.state_dict(), optimizer_path)

        # Save scheduler state
        if scheduler is not None:
            scheduler_path = os.path.join(self.checkpoint_dir, f'scheduler_step_{step}.pth')
            torch.save(scheduler.state_dict(), scheduler_path)

        # Save scaler state (for mixed precision)
        if scaler is not None:
            scaler_path = os.path.join(self.checkpoint_dir, f'scaler_step_{step}.pth')
            torch.save(scaler.state_dict(), scaler_path)

        # Save checkpoint metadata
        checkpoint['model_path'] = model_path
        checkpoint['optimizer_path'] = optimizer_path
        checkpoint['scheduler_path'] = scheduler_path if scheduler is not None else None
        checkpoint['scaler_path'] = scaler_path if scaler is not None else None

        with open(self.checkpoint_path, 'w') as f:
            json.dump(checkpoint, f, indent=2)

        # Clean up old checkpoints (keep last N)
        self._cleanup_old_checkpoints(step)

        print(f"💾 Checkpoint saved at step {step}")
        return checkpoint

    def load_checkpoint(self, model, optimizer=None, scheduler=None, scaler=None):
        """Load latest checkpoint if exists"""

        if not os.path.exists(self.checkpoint_path):
            print("📝 No checkpoint found. Starting from scratch.")
            return {'step': 0, 'epoch': 0, 'best_loss': float('inf')}

        with open(self.checkpoint_path, 'r') as f:
            checkpoint = json.load(f)

        # Load model
        if os.path.exists(checkpoint['model_path']):
            model.load_state_dict(torch.load(checkpoint['model_path'], map_location=device))
            print(f"✅ Model loaded from step {checkpoint['step']}")

        # Load optimizer
        if optimizer and checkpoint.get('optimizer_path') and os.path.exists(checkpoint['optimizer_path']):
            optimizer.load_state_dict(torch.load(checkpoint['optimizer_path'], map_location=device))
            print("✅ Optimizer state loaded")

        # Load scheduler
        if scheduler and checkpoint.get('scheduler_path') and os.path.exists(checkpoint['scheduler_path']):
            scheduler.load_state_dict(torch.load(checkpoint['scheduler_path'], map_location=device))
            print("✅ Scheduler state loaded")

        # Load scaler
        if scaler and checkpoint.get('scaler_path') and os.path.exists(checkpoint['scaler_path']):
            scaler.load_state_dict(torch.load(checkpoint['scaler_path'], map_location=device))
            print("✅ Scaler state loaded")

        print(f"🔄 Resuming from step {checkpoint['step']}, epoch {checkpoint['epoch']}")
        return checkpoint

    def _cleanup_old_checkpoints(self, current_step):
        """Remove old checkpoints, keeping only last N"""

        # Find all checkpoint files
        all_files = os.listdir(self.checkpoint_dir)
        model_files = [f for f in all_files if f.startswith('model_step_')]

        if len(model_files) <= self.config['keep_last_n_checkpoints']:
            return

        # Extract step numbers
        steps = []
        for f in model_files:
            try:
                step = int(f.replace('model_step_', '').replace('.pth', ''))
                steps.append(step)
            except:
                continue

        steps.sort()

        # Remove oldest checkpoints
        to_remove = steps[:-self.config['keep_last_n_checkpoints']]

        for step in to_remove:
            for prefix in ['model', 'optimizer', 'scheduler', 'scaler']:
                path = os.path.join(self.checkpoint_dir, f'{prefix}_step_{step}.pth')
                if os.path.exists(path):
                    os.remove(path)
                    print(f"🗑️  Removed old checkpoint: {prefix}_step_{step}.pth")


# === INITIALIZE CHECKPOINT MANAGER ===
checkpoint_manager = CheckpointManager(CHECKPOINT_DIR, CONFIG)

print("\n✅ Checkpoint manager initialized")
print(f"📁 Checkpoint directory: {CHECKPOINT_DIR}")
print(f"💾 Checkpoints will be saved every {CONFIG['save_every_steps']} steps")
print(f"🔄 Keeping last {CONFIG['keep_last_n_checkpoints']} checkpoints")

# Save config to file for reference
config_path = os.path.join(CHECKPOINT_DIR, 'training_config.json')
with open(config_path, 'w') as f:
    json.dump(CONFIG, f, indent=2)

print(f"\n✅ Configuration saved to: {config_path}")
print("\n⚠️  IMPORTANT: Checkpoints are saved to Google Drive")
print("   They will persist across Colab session disconnections!")

```

Output:

```text

⚙️  Training Configuration:
============================================================
   num_res_blocks                : 15
   num_filters                   : 128
   batch_size                    : 512
   accumulation_steps            : 2
   epochs                        : 10
   total_steps                   : 200000
   learning_rate                 : 0.0001
   weight_decay                  : 0.0001
   betas                         : (0.9, 0.999)
   grad_clip                     : 1.0
   warmup_steps                  : 1000
   lr_schedule                   : onecycle
   loss_value_weight             : 1.0
   loss_policy_weight            : 0.5
   save_every_steps              : 2000
   validate_every_steps          : 5000
   keep_last_n_checkpoints       : 5
   use_amp                       : True
   num_workers                   : 2
   prefetch_factor               : 2
============================================================

✅ Checkpoint manager initialized
📁 Checkpoint directory: /content/drive/MyDrive/GambitFlow_Project/Checkpoints
💾 Checkpoints will be saved every 2000 steps
🔄 Keeping last 5 checkpoints

✅ Configuration saved to: /content/drive/MyDrive/GambitFlow_Project/Checkpoints/training_config.json

⚠️  IMPORTANT: Checkpoints are saved to Google Drive
   They will persist across Colab session disconnections!




```
---

### cell 7
```python



# =-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=
# Cell 7: ULTIMATE TRAINING ENGINE (V11 - FINAL, SCOPE FIX)
# This version corrects the NameError by defining loss functions in the global scope.
# -=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import DataLoader, Dataset
from torch.amp import GradScaler, autocast
import time
from datetime import datetime
import traceback
import numpy as np
from tqdm.auto import tqdm
import sqlite3
import json
import chess
import random
import os

# --- 1. High-Speed RAM-Cached Dataset ---
class FastChessDataset(Dataset):
    def __init__(self, data_cache):
        self.data = data_cache
        print(f"\n🧠 RAM Dataset created with {len(self.data):,} pre-processed samples.")
    def __len__(self): return len(self.data)
    def __getitem__(self, idx): return self.data[idx]

def build_ram_cache(db_paths, weights, total_samples, match_cap):
    print(f"\n🧠 Building RAM Cache... Pre-processing {total_samples:,} samples.")
    print("   🚀 Speed optimized: Fetching data in large batches...")
    cache, conns, db_sizes = [], {}, {}
    for source, path in db_paths.items():
        if not path: continue
        try:
            table = 'puzzles' if source == 'tactical' else 'openings' if source == 'opening' else 'positions' if source == 'match' else 'endgame'
            if isinstance(path, list):
                conns[source] = [sqlite3.connect(f'file:{p}?mode=ro', uri=True) for p in path]
                db_sizes[source] = sum(c.cursor().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for c in conns[source])
            else:
                conn = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
                conns[source] = conn
                db_sizes[source] = conn.cursor().execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        except: pass
    sources_available = [s for s in weights.keys() if s in conns]
    valid_weights = {s: weights[s] for s in sources_available}
    total_w = sum(valid_weights.values())
    samples_per_source = {s: int((w/total_w) * total_samples) for s, w in valid_weights.items()}
    with tqdm(total=total_samples, desc="Caching Data") as pbar:
        for source, count in samples_per_source.items():
            if count <= 0: continue
            BATCH_SIZE = 2000
            try:
                processed = 0
                while processed < count:
                    limit = min(BATCH_SIZE, count - processed)
                    if source == 'opening':
                        rows = conns[source].cursor().execute(f"SELECT fen, move_data FROM openings ORDER BY RANDOM() LIMIT {limit}").fetchall()
                        for fen, move_data_json in rows:
                            try:
                                move_data = json.loads(move_data_json)
                                moves, freqs = list(move_data.keys()), [d.get('frequency', 1) for d in move_data.values()]
                                if not moves: continue
                                selected_move = random.choices(moves, weights=freqs, k=1)[0]
                                score = move_data[selected_move].get('avg_score', 0.5)
                                move_idx = move_to_index(chess.Board(fen).parse_san(selected_move))
                                cache.append((fen_to_28_channels(fen), (score - 0.5) * 2.0, move_idx, source))
                                pbar.update(1)
                            except: continue
                    elif source == 'match':
                        total_rows = min(db_sizes.get('match', 0), match_cap)
                        start_offset = random.randint(0, max(0, total_rows - limit))
                        rows = conns[source].cursor().execute(f"SELECT fen, value_target, move_played FROM positions LIMIT {limit} OFFSET {start_offset}").fetchall()
                        for fen, value, move_san in rows:
                            try:
                                move_idx = move_to_index(chess.Board(fen).parse_san(move_san))
                                cache.append((fen_to_28_channels(fen), value, move_idx, source))
                                pbar.update(1)
                            except: continue
                    elif source == 'tactical':
                        rows = conns[source].cursor().execute(f"SELECT fen, moves FROM puzzles ORDER BY RANDOM() LIMIT {limit}").fetchall()
                        for fen, moves_uci in rows:
                            try:
                                move_idx = move_to_index(chess.Move.from_uci(moves_uci.split()[0]))
                                cache.append((fen_to_28_channels(fen), 1.0, move_idx, source))
                                pbar.update(1)
                            except: continue
                    elif source == 'endgame':
                        conn = random.choice(conns['endgame'])
                        rows = conn.cursor().execute(f"SELECT fen, wdl FROM endgame ORDER BY RANDOM() LIMIT {limit}").fetchall()
                        for fen, wdl in rows:
                            try:
                                cache.append((fen_to_28_channels(fen), float(wdl), 0, source))
                                pbar.update(1)
                            except: continue
                    processed += limit
            except: continue
    for conns_list in conns.values():
        if isinstance(conns_list, list): [c.close() for c in conns_list]
        else: conns_list.close()
    random.shuffle(cache)
    print("📦 Converting to tensors...")
    return [(torch.from_numpy(f), torch.tensor([v]), torch.tensor(m, dtype=torch.long), s) for f, v, m, s in cache]

# --- 2. INITIALIZATION ---
print("🏗️  Initializing training components...")
model = SynapseBase(num_res_blocks=CONFIG['num_res_blocks'], num_filters=CONFIG['num_filters']).to(device)
optimizer = AdamW(model.parameters(), lr=CONFIG['learning_rate'], weight_decay=CONFIG['weight_decay'], betas=CONFIG['betas'])
scheduler = OneCycleLR(optimizer, max_lr=CONFIG['learning_rate'] * 10, total_steps=CONFIG['total_steps'], pct_start=0.1, anneal_strategy='cos')
scaler = GradScaler(enabled=CONFIG['use_amp'])

# --- 3. LOSS FUNCTIONS (FIX: Defined globally) ---
criterion_value = nn.MSELoss()
criterion_policy = nn.CrossEntropyLoss()

# --- 4. LOAD CHECKPOINT ---
checkpoint = checkpoint_manager.load_checkpoint(model, optimizer, scheduler, scaler)
start_step = checkpoint.get('step', 0)
start_epoch = checkpoint.get('epoch', 0)
best_loss = checkpoint.get('best_loss', float('inf'))

# --- 5. BUILD CACHE & DATALOADER ---
SAMPLES_PER_EPOCH = 100_000
data_cache = build_ram_cache(db_paths, SAMPLING_WEIGHTS, SAMPLES_PER_EPOCH, MATCH_DB_CAP)
train_loader = DataLoader(
    FastChessDataset(data_cache), batch_size=CONFIG['batch_size'], num_workers=CONFIG['num_workers'],
    pin_memory=True, shuffle=True, drop_last=True, persistent_workers=True if CONFIG['num_workers'] > 0 else False
)

# --- 6. MAIN TRAINING FUNCTION ---
def execute_training():
    global global_step, start_epoch, best_loss
    print("\n" + "="*60 + f"\n🚀 STARTING FAST TRAINING | Target: {CONFIG['total_steps']:,} | Resuming: {start_step:,}\n" + "="*60)
    model.train()
    for epoch in range(start_epoch, CONFIG['epochs']):
        if global_step >= CONFIG['total_steps']: break
        print(f"\n--- Epoch {epoch+1}/{CONFIG['epochs']} ---")
        if epoch > start_epoch:
            new_cache = build_ram_cache(db_paths, SAMPLING_WEIGHTS, SAMPLES_PER_EPOCH, MATCH_DB_CAP)
            train_loader.dataset.data = new_cache
        pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch+1}")
        for features, values, moves, sources in train_loader:
            if global_step >= CONFIG['total_steps']: break
            features, values, moves = features.to(device, non_blocking=True), values.to(device, non_blocking=True), moves.to(device, non_blocking=True)
            with autocast(device_type='cuda', enabled=CONFIG['use_amp']):
                pred_val, pred_open, pred_tact = model(features)
                loss_val = criterion_value(pred_val, values)
                loss_pol = torch.tensor(0.0, device=device)
                is_opening = torch.tensor([s == 'opening' for s in sources], device=device)
                if is_opening.any():
                    mask = moves[is_opening] < 1968
                    if mask.any(): loss_pol += criterion_policy(pred_open[is_opening][mask], moves[is_opening][mask])
                is_tact_match = torch.tensor([s in ['tactical', 'match'] for s in sources], device=device)
                if is_tact_match.any(): loss_pol += criterion_policy(pred_tact[is_tact_match], moves[is_tact_match])
                total_loss = CONFIG['loss_value_weight'] * loss_val + CONFIG['loss_policy_weight'] * loss_pol
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), CONFIG['grad_clip'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            global_step += 1
            pbar.update(1)
            pbar.set_postfix({'Loss': f'{total_loss.item():.4f}', 'LR': f'{scheduler.get_last_lr()[0]:.2e}'})
            if global_step > 0 and global_step % CONFIG['save_every_steps'] == 0:
                checkpoint_manager.save_checkpoint(model, optimizer, scheduler, scaler, global_step, epoch, best_loss)
        pbar.close()

# --- 7. EXECUTION ---
global_step = start_step
try:
    if global_step < CONFIG['total_steps']: execute_training()
    else: print("🎉 Target steps already reached.")
except KeyboardInterrupt: print("\n🛑 Training paused.")
finally:
    print("\n💾 Saving final state...")
    if 'model' in locals():
        checkpoint_manager.save_checkpoint(model, optimizer, scheduler, scaler, global_step, start_epoch, best_loss)
    print("\n✅ Session concluded.")
```

Output:

```text



 🏗️  Initializing training components...
✅ Model loaded from step 1950
✅ Optimizer state loaded
✅ Scheduler state loaded
✅ Scaler state loaded
🔄 Resuming from step 1950, epoch 0

🧠 Building RAM Cache... Pre-processing 100,000 samples.
   🚀 Speed optimized: Fetching data in large batches...
Caching Data: 100% 100000/100000 [00:24<00:00, 9721.89it/s]📦 Converting to tensors...

🧠 RAM Dataset created with 100,000 pre-processed samples.

============================================================
🚀 STARTING FAST TRAINING | Target: 200,000 | Resuming: 1,950
============================================================

--- Epoch 1/10 ---
Epoch 1: 100% 195/195 [00:21<00:00,  9.75it/s, Loss=3.5598, LR=6.70e-05]💾 Checkpoint saved at step 2000

--- Epoch 2/10 ---

🧠 Building RAM Cache... Pre-processing 100,000 samples.
   🚀 Speed optimized: Fetching data in large batches...
Caching Data: 100% 100000/100000 [00:23<00:00, 11716.56it/s]📦 Converting to tensors...
Epoch 2: 100% 195/195 [00:20<00:00,  9.71it/s, Loss=3.0885, LR=7.21e-05]
--- Epoch 3/10 ---

🧠 Building RAM Cache... Pre-processing 100,000 samples.
   🚀 Speed optimized: Fetching data in large batches...
Caching Data: 100% 100000/100000 [00:22<00:00, 8958.31it/s]📦 Converting to tensors...
Epoch 3: 100% 195/195 [00:20<00:00,  9.76it/s, Loss=2.4420, LR=7.76e-05]
--- Epoch 4/10 ---

🧠 Building RAM Cache... Pre-processing 100,000 samples.
   🚀 Speed optimized: Fetching data in large batches...
Caching Data: 100% 100000/100000 [00:24<00:00, 11540.37it/s]📦 Converting to tensors...
Epoch 4: 100% 195/195 [00:20<00:00,  9.66it/s, Loss=2.3357, LR=8.35e-05]
--- Epoch 5/10 ---

🧠 Building RAM Cache... Pre-processing 100,000 samples.
   🚀 Speed optimized: Fetching data in large batches...
Caching Data: 100% 100000/100000 [00:23<00:00, 9076.38it/s]📦 Converting to tensors...
Epoch 5: 100% 195/195 [00:20<00:00,  9.71it/s, Loss=2.0746, LR=8.98e-05]
--- Epoch 6/10 ---

🧠 Building RAM Cache... Pre-processing 100,000 samples.
   🚀 Speed optimized: Fetching data in large batches...
Caching Data: 100% 100000/100000 [00:24<00:00, 11417.61it/s]📦 Converting to tensors...
Epoch 6: 100% 195/195 [00:20<00:00,  9.68it/s, Loss=1.7066, LR=9.65e-05]
--- Epoch 7/10 ---

🧠 Building RAM Cache... Pre-processing 100,000 samples.
   🚀 Speed optimized: Fetching data in large batches...
Caching Data: 100% 100000/100000 [00:24<00:00, 7650.37it/s]📦 Converting to tensors...
Epoch 7: 100% 195/195 [00:20<00:00,  9.68it/s, Loss=1.4767, LR=1.04e-04]
--- Epoch 8/10 ---

🧠 Building RAM Cache... Pre-processing 100,000 samples.
   🚀 Speed optimized: Fetching data in large batches...
Caching Data: 100% 100000/100000 [00:24<00:00, 10091.54it/s]📦 Converting to tensors...
Epoch 8: 100% 195/195 [00:20<00:00,  9.69it/s, Loss=1.2457, LR=1.11e-04]
--- Epoch 9/10 ---

🧠 Building RAM Cache... Pre-processing 100,000 samples.
   🚀 Speed optimized: Fetching data in large batches...
Caching Data: 100% 100000/100000 [00:24<00:00, 9362.98it/s]📦 Converting to tensors...
Epoch 9: 100% 195/195 [00:20<00:00,  9.75it/s, Loss=1.2912, LR=1.19e-04]
--- Epoch 10/10 ---

🧠 Building RAM Cache... Pre-processing 100,000 samples.
   🚀 Speed optimized: Fetching data in large batches...
Caching Data: 100% 100000/100000 [00:24<00:00, 10562.88it/s]📦 Converting to tensors...
Epoch 10: 100% 195/195 [00:20<00:00,  9.69it/s, Loss=1.1675, LR=1.27e-04]
💾 Saving final state...
💾 Checkpoint saved at step 3900

✅ Session concluded.


```
---

### cell 8
```python

# Cell 8: ONNX Export & Validation
# ==============================================================================
# 📦 EXPORT TO ONNX FOR DEPLOYMENT
# ==============================================================================

# Install ONNX dependencies
print("⚙️ Installing ONNX dependencies...")
!pip install onnx onnxruntime -q

import torch
import onnx
import onnxruntime as ort
import numpy as np
import time
import os
import json

print("📦 Exporting model to ONNX format...")
print("=" * 60)

# === LOAD BEST CHECKPOINT ===
print("\n1️⃣ Loading best checkpoint...")

# Find latest checkpoint
checkpoint_path = os.path.join(CHECKPOINT_DIR, 'checkpoint.json')
with open(checkpoint_path, 'r') as f:
    checkpoint_info = json.load(f)

model_path = checkpoint_info['model_path']
step = checkpoint_info['step']

print(f"   Loading from: {model_path}")
print(f"   Step: {step}")

# Load model
model = SynapseBase(
    num_res_blocks=CONFIG['num_res_blocks'],
    num_filters=CONFIG['num_filters']
)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.eval()
model.to('cpu')  # Export on CPU for compatibility

print("✅ Model loaded successfully")

# === EXPORT TO ONNX ===
print("\n2️⃣ Exporting to ONNX...")

# Dummy input
dummy_input = torch.randn(1, 28, 8, 8, dtype=torch.float32)

# Output path
onnx_path = os.path.join(CHECKPOINT_DIR, 'synapse_base.onnx')

# Export
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=['board'],
    output_names=['value', 'opening_policy', 'tactical_policy'],
    dynamic_axes={
        'board': {0: 'batch_size'},
        'value': {0: 'batch_size'},
        'opening_policy': {0: 'batch_size'},
        'tactical_policy': {0: 'batch_size'}
    }
)

print(f"✅ ONNX model saved: {onnx_path}")

# Check file size
file_size_mb = os.path.getsize(onnx_path) / (1024**2)
print(f"📊 File size: {file_size_mb:.2f} MB")

if file_size_mb > 35:
    print("⚠️  Warning: Model size is larger than expected (target: 20-30 MB)")
else:
    print("✅ Model size is within target range!")

# === VERIFY ONNX MODEL ===
print("\n3️⃣ Verifying ONNX model...")

# Load and check
onnx_model = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model)
print("✅ ONNX model is valid")

# === TEST ONNX RUNTIME INFERENCE ===
print("\n4️⃣ Testing ONNX Runtime inference...")

# Create inference session
session_options = ort.SessionOptions()
session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

session = ort.InferenceSession(onnx_path, session_options, providers=['CPUExecutionProvider'])

print(f"✅ ONNX Runtime session created")
print(f"   Providers: {session.get_providers()}")

# Test inference
test_input = np.random.randn(1, 28, 8, 8).astype(np.float32)

# Warmup
for _ in range(10):
    _ = session.run(None, {'board': test_input})

# Benchmark
num_runs = 100
latencies = []

for _ in range(num_runs):
    start = time.time()
    outputs = session.run(None, {'board': test_input})
    end = time.time()
    latencies.append((end - start) * 1000)  # Convert to ms

value_out, opening_out, tactical_out = outputs

print(f"\n📊 Inference Benchmark (CPU):")
print(f"   Runs: {num_runs}")
print(f"   Mean latency: {np.mean(latencies):.2f} ms")
print(f"   Median latency: {np.median(latencies):.2f} ms")
print(f"   Min latency: {np.min(latencies):.2f} ms")
print(f"   Max latency: {np.max(latencies):.2f} ms")
print(f"   Std dev: {np.std(latencies):.2f} ms")

if np.mean(latencies) < 100:
    print("✅ Inference speed meets target (<100ms)")
else:
    print("⚠️  Inference speed exceeds target (consider optimization)")

# Verify outputs
print(f"\n🔍 Output Shapes:")
print(f"   Value: {value_out.shape}")
print(f"   Opening policy: {opening_out.shape}")
print(f"   Tactical policy: {tactical_out.shape}")

print(f"\n🔍 Output Ranges:")
print(f"   Value: [{value_out.min():.3f}, {value_out.max():.3f}]")
print(f"   Opening logits: [{opening_out.min():.3f}, {opening_out.max():.3f}]")
print(f"   Tactical logits: [{tactical_out.min():.3f}, {tactical_out.max():.3f}]")

# === TEST WITH REAL POSITION ===
print("\n5️⃣ Testing with real chess position...")

# Starting position
test_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
test_features = fen_to_28_channels(test_fen).reshape(1, 28, 8, 8)

# Run inference
value, opening, tactical = session.run(None, {'board': test_features})

print(f"✅ Real position test:")
print(f"   FEN: {test_fen}")
print(f"   Value: {value[0][0]:.6f}")
print(f"   Top opening moves (logits):")

# Get top 5 opening moves
opening_probs = np.exp(opening[0]) / np.sum(np.exp(opening[0]))
top_5_opening = np.argsort(opening_probs)[-5:][::-1]
for idx in top_5_opening:
    print(f"      Move {idx}: {opening_probs[idx]:.4f}")

print(f"   Top tactical move (index): {np.argmax(tactical[0])}")

# === SAVE PYTORCH MODEL TOO ===
print("\n6️⃣ Saving PyTorch model for fine-tuning...")

pth_path = os.path.join(CHECKPOINT_DIR, 'synapse_base.pth')
torch.save(model.state_dict(), pth_path)

print(f"✅ PyTorch model saved: {pth_path}")

# === SUMMARY ===
print("\n" + "=" * 60)
print("✅ EXPORT COMPLETE!")
print("=" * 60)
print(f"📦 ONNX model: {onnx_path}")
print(f"📦 PyTorch model: {pth_path}")
print(f"📊 Model size: {file_size_mb:.2f} MB")
print(f"⚡ Inference time: {np.mean(latencies):.2f} ms")
print("\n🎯 Files ready for upload to HuggingFace:")
print(f"   1. {os.path.basename(onnx_path)}")
print(f"   2. {os.path.basename(pth_path)}")
print("\n📝 Next steps:")
print("   1. Upload both files to HuggingFace (GambitFlow/Synapse-Base)")
print("   2. Set up self-play workers in HF Spaces")
print("   3. Start continuous improvement loop")
print("=" * 60)


```

Output:

```text


 ⚙️ Installing ONNX dependencies...
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.5/17.5 MB 123.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 17.4/17.4 MB 115.2 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 46.0/46.0 kB 4.1 MB/s eta 0:00:00
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 86.8/86.8 kB 9.2 MB/s eta 0:00:00
📦 Exporting model to ONNX format...
============================================================

1️⃣ Loading best checkpoint...
   Loading from: /content/drive/MyDrive/GambitFlow_Project/Checkpoints/model_step_3900.pth
   Step: 3900
✅ Model loaded successfully

2️⃣ Exporting to ONNX...
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipython-input-2887437333.py in <cell line: 0>()
     54 
     55 # Export
---> 56 torch.onnx.export(
     57     model,
     58     dummy_input,

2 frames/usr/local/lib/python3.12/dist-packages/torch/onnx/__init__.py in export(model, args, f, kwargs, verbose, input_names, output_names, opset_version, dynamo, external_data, dynamic_shapes, custom_translation_table, report, optimize, verify, profile, dump_exported_program, artifacts_dir, fallback, export_params, keep_initializers_as_inputs, dynamic_axes, training, operator_export_type, do_constant_folding, custom_opsets, export_modules_as_functions, autograd_inlining)
    280     """
    281     if dynamo is True or isinstance(model, torch.export.ExportedProgram):
--> 282         from torch.onnx._internal.exporter import _compat
    283 
    284         if isinstance(args, torch.Tensor):

/usr/local/lib/python3.12/dist-packages/torch/onnx/_internal/exporter/_compat.py in <module>
     14 from torch.onnx import _constants as onnx_constants
     15 from torch.onnx._internal._lazy_import import onnx, onnxscript_apis, onnxscript_ir as ir
---> 16 from torch.onnx._internal.exporter import (
     17     _constants,
     18     _core,

/usr/local/lib/python3.12/dist-packages/torch/onnx/_internal/exporter/_core.py in <module>
     16 from typing import Any, Callable, Literal
     17 
---> 18 import onnxscript
     19 import onnxscript.evaluator
     20 from onnxscript import ir

ModuleNotFoundError: No module named 'onnxscript'

---------------------------------------------------------------------------
NOTE: If your import is failing due to a missing package, you can
manually install dependencies using either !pip or !apt.

To view examples of installing some common dependencies, click the
"Open Examples" button below.
---------------------------------------------------------------------------
Open Examples


```
---
