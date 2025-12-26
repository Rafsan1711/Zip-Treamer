## Notebook Name: Synapse_Edge_01_DataPrep.ipynb

---

## 1. Introduction
eta amar  GambitFlow/Synapse-Edge  model er dataset bananor notebook. 

---

### Cell 1
```python
# Cell 1: Smart Drive Checkpoint Management (10GB Limit)
# Compatible: GPU ONLY
# Strategy: Keep only essential checkpoints

import os
import torch
import json
from datetime import datetime
from google.colab import drive
import glob

# Mount Drive
drive.mount('/content/drive')

# Paths
BASE_PATH = '/content/drive/MyDrive/Chessmate_Project/Synapse_Edge'
PATHS = {
    'checkpoints': os.path.join(BASE_PATH, 'checkpoints'),
    'data': os.path.join(BASE_PATH, 'data'),
    'models': os.path.join(BASE_PATH, 'models'),
    'logs': os.path.join(BASE_PATH, 'logs')
}

for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

# GPU Check
if not torch.cuda.is_available():
    raise RuntimeError("❌ NO GPU!")

device = torch.device("cuda")
gpu_name = torch.cuda.get_device_name(0)
gpu_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)

print(f"{'='*60}")
print(f"🚀 SYNAPSE-EDGE TRAINING (SMART STORAGE)")
print(f"{'='*60}")
print(f"🖥️  GPU: {gpu_name}")
print(f"💾 VRAM: {gpu_memory:.2f} GB")
print(f"{'='*60}\n")


# Smart Checkpoint Manager
class SmartCheckpointManager:
    """
    Storage Strategy (Fits in 10GB):
    - Keep ONLY 1 checkpoint per worker (resume)
    - Keep ONLY best model (fine-tuning)
    - Auto-delete old checkpoints
    """

    def __init__(self, worker_id):
        self.worker_id = worker_id
        self.checkpoint_dir = PATHS['checkpoints']
        self.state_file = os.path.join(self.checkpoint_dir, f'worker_{worker_id}_state.json')
        self.checkpoint_file = os.path.join(self.checkpoint_dir, f'worker_{worker_id}_latest.pt')
        self.best_model_file = os.path.join(self.checkpoint_dir, 'best_model.pt')
        self.best_finetune_file = os.path.join(self.checkpoint_dir, 'best_model_finetune.pt')

    def load_state(self):
        """Load training state"""
        if os.path.exists(self.state_file):
            with open(self.state_file, 'r') as f:
                state = json.load(f)
            print(f"📂 Worker {self.worker_id} Resuming:")
            print(f"   - Epoch: {state.get('epoch', 0)}")
            print(f"   - Steps: {state.get('total_steps', 0):,}")
            print(f"   - Best Loss: {state.get('best_loss', float('inf')):.6f}")
            return state

        return {
            'epoch': 0,
            'total_steps': 0,
            'best_loss': float('inf'),
            'worker_id': self.worker_id, # FIX: Changed worker_id to self.worker_id
            'training_start': datetime.now().isoformat()
        }

    def save_state(self, state):
        """Save training state (lightweight JSON)"""
        state['last_updated'] = datetime.now().isoformat()
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)

    def save_checkpoint(self, model, optimizer, scaler, state):
        """
        Save checkpoint:
        - This worker's latest → worker_X_latest.pt (for resume)
        - If best across all workers → best_model.pt (for deployment)
        """

        # 1. Save this worker's checkpoint (ALWAYS)
        checkpoint_data = {
            'epoch': state['epoch'],
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'loss': state.get('current_loss', 0),
            'best_loss': state['best_loss'],
            'total_steps': state['total_steps'],
            'worker_id': self.worker_id
        }

        torch.save(checkpoint_data, self.checkpoint_file)
        print(f"💾 Worker {self.worker_id} checkpoint saved")

        # 2. Check if this is the BEST across all workers
        current_loss = state.get('current_loss', float('inf'))
        global_best_loss = self._get_global_best_loss()

        if current_loss < global_best_loss:
            print(f"\n🏆 NEW GLOBAL BEST! Loss: {current_loss:.6f}")

            # Save best model (for deployment)
            torch.save(checkpoint_data, self.best_model_file)

            # Save fine-tuning version
            model.save_for_finetuning(self.best_finetune_file)

            print(f"💎 Best model updated: {os.path.basename(self.best_model_file)}")

        # 3. Clean up old checkpoints from other workers (if storage low)
        self._cleanup_if_needed()

    def _get_global_best_loss(self):
        """Check best loss from all workers"""
        if os.path.exists(self.best_model_file):
            checkpoint = torch.load(self.best_model_file, map_location='cpu')
            return checkpoint.get('loss', float('inf'))
        return float('inf')

    def _cleanup_if_needed(self):
        """Clean up if storage > 8GB"""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, '*.pt'))

        # Calculate total size
        total_size = sum(os.path.getsize(f) for f in checkpoint_files) / (1024**3)

        if total_size > 8.0:  # If > 8GB
            print(f"⚠️  Storage: {total_size:.2f} GB. Cleaning up...")

            # Keep only:
            # - This worker's latest
            # - Best model
            # - Best finetune
            keep_files = {
                self.checkpoint_file,
                self.best_model_file,
                self.best_finetune_file
            }

            for file in checkpoint_files:
                if file not in keep_files:
                    try:
                        os.remove(file)
                        print(f"   Deleted: {os.path.basename(file)}")
                    except:
                        pass

    def load_checkpoint(self, model, optimizer, scaler):
        """Load checkpoint for this worker"""

        if os.path.exists(self.checkpoint_file):
            print(f"📥 Loading Worker {self.worker_id} checkpoint...")

            try:
                checkpoint = torch.load(self.checkpoint_file, map_location=device)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                scaler.load_state_dict(checkpoint['scaler_state_dict'])

                print(f"✅ Loaded successfully")
                print(f"   - Epoch: {checkpoint['epoch']}")
                print(f"   - Steps: {checkpoint.get('total_steps', 0):,}")

                return checkpoint

            except Exception as e:
                print(f"⚠️  Load failed: {e}")
                return None
        else:
            print(f"ℹ️  No checkpoint for Worker {self.worker_id}. Starting fresh.")
            return None

    def get_storage_info(self):
        """Display storage usage"""
        checkpoint_files = glob.glob(os.path.join(self.checkpoint_dir, '*.pt'))

        if checkpoint_files:
            total_size = sum(os.path.getsize(f) for f in checkpoint_files) / (1024**3)
            print(f"\n💾 Storage Info:")
            print(f"   Total checkpoint size: {total_size:.2f} GB")
            print(f"   Files: {len(checkpoint_files)}")
            for f in checkpoint_files:
                size = os.path.getsize(f) / (1024**2)
                print(f"   - {os.path.basename(f)}: {size:.1f} MB")


# Get worker_id from environment or config
# This will be set in Cell 4 with ACCOUNT_CONFIG
WORKER_ID = 0  # Will be updated by Cell 4

# Initialize (actual initialization happens after Cell 4)
print("✅ Smart checkpoint system ready!")
print(f"💾 Max storage: ~9 GB (fits in 10GB limit)")
print(f"📦 Strategy: Keep 1 checkpoint per worker + 1 best model\n")

```


Output:



```text
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
============================================================
🚀 SYNAPSE-EDGE TRAINING (SMART STORAGE)
============================================================
🖥️  GPU: Tesla T4
💾 VRAM: 14.74 GB
============================================================

✅ Smart checkpoint system ready!
💾 Max storage: ~9 GB (fits in 10GB limit)
📦 Strategy: Keep 1 checkpoint per worker + 1 best model

```

---

### Cell 2
```python
# Cell 2: Install Training Dependencies
# Compatible: GPU ONLY

print("📦 Installing Training Libraries...")

# PyTorch (already installed in Colab, but verify version)
import torch
print(f"✅ PyTorch {torch.__version__}")

# Additional libraries
!pip install -q timm  # PyTorch Image Models (advanced architectures)
!pip install -q tensorboard  # Training visualization

import timm
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import IterableDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter

print("✅ All training dependencies ready!")

```


Output:



```text
📦 Installing Training Libraries...
✅ PyTorch 2.9.0+cu126
✅ All training dependencies ready!
```

---

### Cell 3
```python


# Cell 3: Model Architecture (FIXED for BCEWithLogitsLoss)
# Compatible: GPU ONLY
# Fix: Remove Sigmoid from tactical head

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==================== COMPONENTS (Same as before) ====================

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c)
        y = self.excitation(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.spatial = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        channel_att = torch.sigmoid(avg_out + max_out)
        x = x * channel_att

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        spatial_att = self.spatial(torch.cat([avg_out, max_out], dim=1))
        x = x * spatial_att

        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels, use_attention=True):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)
        self.attention = CBAM(channels) if use_attention else None
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.attention:
            out = self.attention(out)

        out += identity
        return self.relu(out)


class MultiScaleFeature(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        branch_channels = out_channels // 4

        self.branch1 = nn.Conv2d(in_channels, branch_channels, 1)
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.Conv2d(branch_channels, branch_channels, 3, padding=1)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, branch_channels, 1),
            nn.Conv2d(branch_channels, branch_channels, 5, padding=2)
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, branch_channels, 1)
        )

    def forward(self, x):
        return torch.cat([
            self.branch1(x), self.branch2(x),
            self.branch3(x), self.branch4(x)
        ], dim=1)


# ==================== MAIN MODEL (FIXED) ====================

class SynapseEdge(nn.Module):
    """Synapse-Edge with BCEWithLogitsLoss compatibility"""

    def __init__(
        self,
        num_residual_blocks=20,
        base_channels=192,
        use_attention=True,
        pretrained_path=None
    ):
        super().__init__()

        self.model_version = "v3.0"
        self.num_residual_blocks = num_residual_blocks
        self.base_channels = base_channels

        # Input
        self.input_conv = nn.Sequential(
            nn.Conv2d(12, base_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(inplace=True)
        )
        self.multiscale = MultiScaleFeature(base_channels, base_channels)

        # Backbone
        blocks_per_group = num_residual_blocks // 4

        self.backbone_group1 = nn.ModuleList([
            ResidualBlock(base_channels, use_attention)
            for _ in range(blocks_per_group)
        ])
        self.backbone_group2 = nn.ModuleList([
            ResidualBlock(base_channels, use_attention)
            for _ in range(blocks_per_group)
        ])
        self.backbone_group3 = nn.ModuleList([
            ResidualBlock(base_channels, use_attention)
            for _ in range(blocks_per_group)
        ])
        self.backbone_group4 = nn.ModuleList([
            ResidualBlock(base_channels, use_attention)
            for _ in range(num_residual_blocks - 3*blocks_per_group)
        ])

        # Heads
        self.policy_head = nn.Sequential(
            nn.Conv2d(base_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(64 * 64, 2048),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(2048, 1968)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(base_channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 1)
        )

        # FIX: Remove Sigmoid (BCEWithLogitsLoss does it internally)
        self.tactical_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(base_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1)
            # Sigmoid removed!
        )

        self._initialize_weights()

        if pretrained_path:
            self.load_pretrained(pretrained_path)

    def _initialize_weights(self):
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
        x = self.input_conv(x)
        x = self.multiscale(x)

        for block in self.backbone_group1:
            x = block(x)
        for block in self.backbone_group2:
            x = block(x)
        for block in self.backbone_group3:
            x = block(x)
        for block in self.backbone_group4:
            x = block(x)

        policy = self.policy_head(x)
        value = torch.tanh(self.value_head(x))
        tactical = self.tactical_head(x)  # Returns logits, not probabilities

        return policy, value, tactical

    # Fine-tuning methods (same as before)
    def freeze_backbone(self):
        print("🔒 Freezing backbone...")
        for param in self.input_conv.parameters():
            param.requires_grad = False
        for param in self.multiscale.parameters():
            param.requires_grad = False
        for group in [self.backbone_group1, self.backbone_group2,
                      self.backbone_group3, self.backbone_group4]:
            for block in group:
                for param in block.parameters():
                    param.requires_grad = False
        print("✅ Backbone frozen")

    def unfreeze_backbone(self):
        print("🔓 Unfreezing all layers...")
        for param in self.parameters():
            param.requires_grad = True
        print("✅ All layers trainable")

    def progressive_unfreeze(self, stage):
        self.freeze_backbone()

        if stage >= 1:
            for param in self.backbone_group4.parameters():
                param.requires_grad = True
            print("✅ Stage 1: Group 4 unfrozen")
        if stage >= 2:
            for param in self.backbone_group3.parameters():
                param.requires_grad = True
            print("✅ Stage 2: Group 3 unfrozen")
        if stage >= 3:
            for param in self.backbone_group2.parameters():
                param.requires_grad = True
            print("✅ Stage 3: Group 2 unfrozen")
        if stage >= 4:
            self.unfreeze_backbone()
            print("✅ Stage 4: All unfrozen")

    def get_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_for_finetuning(self, path):
        torch.save({
            'model_version': self.model_version,
            'state_dict': self.state_dict(),
            'architecture': {
                'num_residual_blocks': self.num_residual_blocks,
                'base_channels': self.base_channels
            },
            'training_metadata': {
                'framework': 'pytorch',
                'finetunable': True
            }
        }, path)
        print(f"💾 Model saved for fine-tuning: {path}")

    def load_pretrained(self, path, strict=True):
        print(f"📥 Loading pretrained weights: {path}")
        checkpoint = torch.load(path, map_location='cpu')

        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        else:
            state_dict = checkpoint

        self.load_state_dict(state_dict, strict=strict)
        print("✅ Pretrained weights loaded")


# Create model
def create_synapse_edge(device='cuda', pretrained_path=None, freeze_backbone=False):
    model = SynapseEdge(
        num_residual_blocks=20,
        base_channels=192,
        use_attention=True,
        pretrained_path=pretrained_path
    ).to(device)

    if freeze_backbone:
        model.freeze_backbone()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = model.get_trainable_params()

    print(f"{'='*60}")
    print(f"🧠 SYNAPSE-EDGE MODEL (BCEWithLogits Compatible)")
    print(f"{'='*60}")
    print(f"📊 Total Parameters: {total_params:,}")
    print(f"🔧 Trainable Parameters: {trainable_params:,}")
    print(f"💾 Model Size (FP32): {total_params * 4 / (1024**2):.2f} MB")
    print(f"⚡ Model Size (Quantized): {total_params / (1024**2):.2f} MB")
    print(f"{'='*60}\n")

    return model


# Instantiate
PRETRAINED_CHECKPOINT = None
model = create_synapse_edge(device=device, pretrained_path=PRETRAINED_CHECKPOINT, freeze_backbone=False)

print("✅ Model ready!")

```


Output:



```text

============================================================
🧠 SYNAPSE-EDGE MODEL (BCEWithLogits Compatible)
============================================================
📊 Total Parameters: 27,086,138
🔧 Trainable Parameters: 27,086,138
💾 Model Size (FP32): 103.33 MB
⚡ Model Size (Quantized): 25.83 MB
============================================================

✅ Model ready!
  
```

---

### Cell 4
```python
# Cell 4: Data Loader + Checkpoint Initialization
# Compatible: GPU ONLY

import torch
from torch.utils.data import IterableDataset, DataLoader
import sqlite3
import json
import numpy as np
import random
import os

# FEN to Tensor
def fen_to_tensor(fen):
    position = fen.split(' ')[0]
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    piece_to_channel = {
        'P':0, 'N':1, 'B':2, 'R':3, 'Q':4, 'K':5,
        'p':6, 'n':7, 'b':8, 'r':9, 'q':10, 'k':11
    }

    rank = 0
    file = 0

    for char in position:
        if char == '/':
            rank += 1
            file = 0
        elif char.isdigit():
            file += int(char)
        elif char in piece_to_channel:
            tensor[piece_to_channel[char], rank, file] = 1.0
            file += 1

    return torch.from_numpy(tensor)


def move_to_index(move_san):
    return hash(move_san) % 1968


# Dataset with Modulo Partitioning
class DistributedChessDataset(IterableDataset):

    def __init__(self, db_path, worker_id=0, num_workers=1, shuffle_buffer=50000):
        self.db_path = db_path
        self.worker_id = worker_id
        self.num_workers = num_workers
        self.shuffle_buffer = shuffle_buffer

        # Detect schema
        with sqlite3.connect(db_path) as conn:
            cursor = conn.cursor()

            cursor.execute("PRAGMA table_info(training_data)")
            columns = [row[1] for row in cursor.fetchall()]
            self.has_is_tactical = 'is_tactical' in columns

            cursor.execute("SELECT COUNT(*) FROM training_data WHERE best_move IS NOT NULL")
            total_valid_rows = cursor.fetchone()[0]

        self.total_valid_rows = total_valid_rows
        self._estimated_length = total_valid_rows // num_workers

        print(f"📊 Worker {worker_id}/{num_workers}")
        print(f"   Total valid rows: {total_valid_rows:,}")
        print(f"   Per worker: ~{self._estimated_length:,}")

    def __len__(self):
        return self._estimated_length

    def __iter__(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        select_columns = ['fen', 'position_stats', 'best_move']
        if self.has_is_tactical:
            select_columns.append('is_tactical')

        # Modulo partitioning
        query = f'''
            SELECT {', '.join(select_columns)}
            FROM training_data
            WHERE best_move IS NOT NULL
            AND (ROWID % {self.num_workers}) = {self.worker_id}
        '''

        cursor.execute(query)

        buffer = []

        for row in cursor:
            if self.has_is_tactical:
                fen, stats_json, best_move, is_tactical = row
            else:
                fen, stats_json, best_move = row
                is_tactical = 0

            try:
                stats = json.loads(stats_json)
                total = stats.get('total', 1)

                if total > 0:
                    moves_stats = stats.get('moves', {}).get(best_move, {})
                    white_wins = moves_stats.get('white', 0)
                    black_wins = moves_stats.get('black', 0)
                    value = (white_wins - black_wins) / total
                else:
                    value = 0.0

                buffer.append({
                    'fen': fen,
                    'best_move': best_move,
                    'value': value,
                    'is_tactical': is_tactical
                })

                if len(buffer) >= self.shuffle_buffer:
                    random.shuffle(buffer)
                    for item in buffer[:self.shuffle_buffer // 2]:
                        yield self._process_sample(item)
                    buffer = buffer[self.shuffle_buffer // 2:]
            except:
                continue

        random.shuffle(buffer)
        for item in buffer:
            yield self._process_sample(item)

        conn.close()

    def _process_sample(self, sample):
        board_tensor = fen_to_tensor(sample['fen'])
        move_idx = move_to_index(sample['best_move'])
        policy_target = torch.tensor(move_idx, dtype=torch.long)
        value_target = torch.tensor([sample['value']], dtype=torch.float32)
        tactical_target = torch.tensor([sample['is_tactical']], dtype=torch.float32)

        return board_tensor, policy_target, value_target, tactical_target


# ==================== CONFIGURATION ====================

ACCOUNT_CONFIG = {
    'worker_id': 2,        # ⚠️ CHANGE THIS: 0, 1, 2
    'num_workers': 3,
    'batch_size': 256,
    'num_dataloader_workers': 4
}

print(f"\n{'='*60}")
print(f"🔧 ACCOUNT CONFIGURATION")
print(f"{'='*60}")
print(f"📍 Worker ID: {ACCOUNT_CONFIG['worker_id']}")
print(f"👥 Total Workers: {ACCOUNT_CONFIG['num_workers']}")
print(f"📦 Batch Size: {ACCOUNT_CONFIG['batch_size']}")
print(f"{'='*60}\n")

# Verify
if ACCOUNT_CONFIG['worker_id'] >= ACCOUNT_CONFIG['num_workers']:
    raise ValueError(f"❌ worker_id must be < num_workers!")

# Initialize Checkpoint Manager NOW (with correct worker_id)
train_checkpoint = SmartCheckpointManager(worker_id=ACCOUNT_CONFIG['worker_id'])
training_state = train_checkpoint.load_state()

# Show storage info
train_checkpoint.get_storage_info()

# Load database
final_db = os.path.join(PATHS['data'], 'synapse_training_final.db')

if not os.path.exists(final_db):
    raise FileNotFoundError(f"❌ Database not found")

print(f"\n📂 Database: {os.path.basename(final_db)}")
print(f"💾 Size: {os.path.getsize(final_db) / (1024**3):.2f} GB\n")

# Create dataset
train_dataset = DistributedChessDataset(
    db_path=final_db,
    worker_id=ACCOUNT_CONFIG['worker_id'],
    num_workers=ACCOUNT_CONFIG['num_workers'],
    shuffle_buffer=50000
)

# DataLoader
train_loader = DataLoader(
    train_dataset,
    batch_size=ACCOUNT_CONFIG['batch_size'],
    num_workers=ACCOUNT_CONFIG['num_dataloader_workers'],
    pin_memory=True,
    persistent_workers=True
)

print("✅ Data loader ready!")
print(f"📊 Positions: ~{len(train_dataset):,}")
print(f"📦 Batches: ~{len(train_dataset) // ACCOUNT_CONFIG['batch_size']:,}\n")
```


Output:



```text

============================================================
🔧 ACCOUNT CONFIGURATION
============================================================
📍 Worker ID: 2
👥 Total Workers: 3
📦 Batch Size: 256
============================================================


💾 Storage Info:
   Total checkpoint size: 0.30 GB
   Files: 2
   - worker_1_latest.pt: 310.3 MB
   - worker_0_latest.pt: 0.0 MB

📂 Database: synapse_training_final.db
💾 Size: 1.05 GB

📊 Worker 2/3
   Total valid rows: 5,551,558
   Per worker: ~1,850,519
✅ Data loader ready!
📊 Positions: ~1,850,519
📦 Batches: ~7,228

              
         
```


### Cell 5

```python

# Cell 5: Multi-Task Loss Function (FIXED for Autocast)
# Compatible: GPU ONLY
# Fix: Use BCEWithLogitsLoss instead of BCELoss

import torch
import torch.nn as nn
import torch.nn.functional as F

class SynapseEdgeLoss(nn.Module):
    """
    Multi-task loss for Synapse-Edge (Autocast Compatible)

    Components:
    1. Policy Loss (CrossEntropy) - Learn best moves
    2. Value Loss (MSE) - Learn position evaluation
    3. Tactical Loss (BCEWithLogits) - Learn tactical recognition
    4. Regularization (L2) - Prevent overfitting
    """

    def __init__(
        self,
        policy_weight=1.0,
        value_weight=0.5,
        tactical_weight=0.3,
        l2_weight=1e-5
    ):
        super().__init__()

        self.policy_weight = policy_weight
        self.value_weight = value_weight
        self.tactical_weight = tactical_weight
        self.l2_weight = l2_weight

        # Loss functions
        self.policy_loss_fn = nn.CrossEntropyLoss()
        self.value_loss_fn = nn.MSELoss()

        # FIX: Use BCEWithLogitsLoss (autocast safe)
        self.tactical_loss_fn = nn.BCEWithLogitsLoss()

    def forward(self, outputs, targets, model=None):
        """
        Args:
            outputs: (policy, value, tactical) from model
            targets: (policy_target, value_target, tactical_target)
            model: For L2 regularization

        Returns:
            total_loss, loss_dict
        """
        policy_pred, value_pred, tactical_pred = outputs
        policy_target, value_target, tactical_target = targets

        # 1. Policy Loss
        policy_loss = self.policy_loss_fn(policy_pred, policy_target)

        # 2. Value Loss
        value_loss = self.value_loss_fn(value_pred, value_target)

        # 3. Tactical Loss (FIX: BCEWithLogitsLoss expects logits, not probabilities)
        tactical_loss = self.tactical_loss_fn(tactical_pred, tactical_target)

        # 4. L2 Regularization
        l2_loss = 0.0
        if model is not None:
            for param in model.parameters():
                l2_loss += torch.norm(param, 2)

        # Combined loss
        total_loss = (
            self.policy_weight * policy_loss +
            self.value_weight * value_loss +
            self.tactical_weight * tactical_loss +
            self.l2_weight * l2_loss
        )

        # Loss breakdown
        loss_dict = {
            'total': total_loss.item(),
            'policy': policy_loss.item(),
            'value': value_loss.item(),
            'tactical': tactical_loss.item(),
            'l2': l2_loss.item() if isinstance(l2_loss, torch.Tensor) else l2_loss
        }

        return total_loss, loss_dict


# Initialize loss
criterion = SynapseEdgeLoss(
    policy_weight=1.0,
    value_weight=0.5,
    tactical_weight=0.3,
    l2_weight=1e-5
)

print("✅ Multi-task loss function ready (Autocast Compatible)!")
print("\n📊 Loss Components:")
print(f"   🎯 Policy Loss: CrossEntropyLoss")
print(f"   ♟️  Value Loss: MSELoss")
print(f"   ⚡ Tactical Loss: BCEWithLogitsLoss (autocast safe)")
print(f"   🛡️  L2 Regularization: {criterion.l2_weight}")
```

Output:



```text
✅ Multi-task loss function ready (Autocast Compatible)!

📊 Loss Components:
   🎯 Policy Loss: CrossEntropyLoss
   ♟️  Value Loss: MSELoss
   ⚡ Tactical Loss: BCEWithLogitsLoss (autocast safe)
   🛡️  L2 Regularization: 1e-05

```

### Cell 6
```python
# Cell 6: Training Loop (Save to Drive Every Epoch)
# Compatible: GPU ONLY

import torch
import time
from tqdm import tqdm

# Training Config
TRAINING_CONFIG = {
    'epochs': 50,
    'gradient_accumulation': 4,
    'learning_rate': 0.001,
    'weight_decay': 1e-5,
    'warmup_epochs': 3,
    'log_every_n_steps': 100,
    'max_grad_norm': 1.0,
    'early_stopping_patience': 10
}

print(f"{'='*60}")
print(f"🎯 TRAINING CONFIGURATION")
print(f"{'='*60}")
print(f"📍 Worker: {ACCOUNT_CONFIG['worker_id']}/{ACCOUNT_CONFIG['num_workers']}")
print(f"🔄 Epochs: {TRAINING_CONFIG['epochs']}")
print(f"💾 Checkpoint: Every epoch (to Drive)")
print(f"{'='*60}\n")

# Optimizer
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=TRAINING_CONFIG['learning_rate'],
    weight_decay=TRAINING_CONFIG['weight_decay']
)

# Scheduler
dataset_size = len(train_dataset)
batch_size = ACCOUNT_CONFIG['batch_size']
gradient_accumulation = TRAINING_CONFIG['gradient_accumulation']

batches_per_epoch = dataset_size // batch_size
steps_per_epoch = batches_per_epoch // gradient_accumulation

print(f"📊 Training Math:")
print(f"   Dataset: {dataset_size:,} positions")
print(f"   Batches/epoch: {batches_per_epoch:,}")
print(f"   Steps/epoch: {steps_per_epoch:,}")
print(f"   Total steps: {steps_per_epoch * TRAINING_CONFIG['epochs']:,}\n")

from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer,
    max_lr=TRAINING_CONFIG['learning_rate'],
    epochs=TRAINING_CONFIG['epochs'],
    steps_per_epoch=steps_per_epoch,
    pct_start=TRAINING_CONFIG['warmup_epochs'] / TRAINING_CONFIG['epochs'],
    anneal_strategy='cos'
)

scaler = torch.amp.GradScaler('cuda')

# Resume
start_epoch = training_state['epoch']
total_steps = training_state['total_steps']
best_loss = training_state['best_loss']
early_stop_counter = 0

checkpoint_data = train_checkpoint.load_checkpoint(model, optimizer, scaler)
if checkpoint_data:
    start_epoch = checkpoint_data['epoch']
    best_loss = checkpoint_data.get('best_loss', float('inf'))

# Training Loop
def train_one_epoch(epoch):
    global total_steps, best_loss, early_stop_counter

    model.train()
    epoch_loss = 0.0
    epoch_metrics = {'policy': 0.0, 'value': 0.0, 'tactical': 0.0}

    optimizer.zero_grad()

    pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{TRAINING_CONFIG['epochs']}")

    for batch_idx, (board, policy_target, value_target, tactical_target) in enumerate(pbar):

        board = board.to(device)
        policy_target = policy_target.to(device)
        value_target = value_target.to(device)
        tactical_target = tactical_target.to(device)

        with torch.amp.autocast('cuda'):
            policy_pred, value_pred, tactical_pred = model(board)
            loss, loss_dict = criterion(
                (policy_pred, value_pred, tactical_pred),
                (policy_target, value_target, tactical_target),
                model
            )
            loss = loss / TRAINING_CONFIG['gradient_accumulation']

        scaler.scale(loss).backward()

        if (batch_idx + 1) % TRAINING_CONFIG['gradient_accumulation'] == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), TRAINING_CONFIG['max_grad_norm'])
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            scheduler.step()
            total_steps += 1

            if total_steps % TRAINING_CONFIG['log_every_n_steps'] == 0:
                pbar.set_postfix({'loss': loss_dict['total'], 'lr': scheduler.get_last_lr()[0]})

        epoch_loss += loss_dict['total']
        epoch_metrics['policy'] += loss_dict['policy']
        epoch_metrics['value'] += loss_dict['value']
        epoch_metrics['tactical'] += loss_dict['tactical']

    num_batches = len(train_loader)
    avg_loss = epoch_loss / num_batches

    if avg_loss >= best_loss:
        early_stop_counter += 1
    else:
        early_stop_counter = 0

    print(f"\n{'='*60}")
    print(f"📊 Epoch {epoch} Summary (Worker {ACCOUNT_CONFIG['worker_id']})")
    print(f"{'='*60}")
    print(f"   Loss: {avg_loss:.6f}")
    print(f"   Policy: {epoch_metrics['policy'] / num_batches:.6f}")
    print(f"   Value: {epoch_metrics['value'] / num_batches:.6f}")
    print(f"   Tactical: {epoch_metrics['tactical'] / num_batches:.6f}")
    print(f"   LR: {scheduler.get_last_lr()[0]:.6f}")
    print(f"   Early Stop: {early_stop_counter}/{TRAINING_CONFIG['early_stopping_patience']}")
    print(f"{'='*60}\n")

    return avg_loss

# Main Training
print(f"{'='*60}")
print(f"🚀 STARTING TRAINING")
print(f"{'='*60}")
print(f"📍 Worker: {ACCOUNT_CONFIG['worker_id']}")
print(f"🎯 Starting Epoch: {start_epoch + 1}")
print(f"🏆 Best Loss: {best_loss:.6f}")
print(f"{'='*60}\n")

try:
    for epoch in range(start_epoch + 1, TRAINING_CONFIG['epochs'] + 1):

        epoch_start = time.time()
        avg_loss = train_one_epoch(epoch)
        epoch_time = time.time() - epoch_start

        # Update state
        training_state['epoch'] = epoch
        training_state['total_steps'] = total_steps
        training_state['current_loss'] = avg_loss

        if avg_loss < best_loss:
            best_loss = avg_loss
            training_state['best_loss'] = best_loss

        # Save checkpoint (to Drive, every epoch)
        print(f"💾 Saving checkpoint to Drive...")
        train_checkpoint.save_checkpoint(model, optimizer, scaler, training_state)
        train_checkpoint.save_state(training_state)

        print(f"⏱️  Epoch time: {epoch_time/60:.2f} min")
        print(f"💾 Progress saved to Drive ✅\n")

        if early_stop_counter >= TRAINING_CONFIG['early_stopping_patience']:
            print(f"⚠️  Early stopping triggered!")
            break

except KeyboardInterrupt:
    print("\n⚠️  Training interrupted by user!")
    print("💾 Saving final checkpoint...")
    training_state['interrupted'] = True
    train_checkpoint.save_checkpoint(model, optimizer, scaler, training_state)
    train_checkpoint.save_state(training_state)
    print("✅ Progress saved. You can resume later!")

except Exception as e:
    print(f"\n❌ Training error: {e}")
    print("💾 Saving emergency checkpoint...")
    training_state['error'] = str(e)
    train_checkpoint.save_checkpoint(model, optimizer, scaler, training_state)
    train_checkpoint.save_state(training_state)
    raise e

# Final summary
print(f"\n{'='*60}")
print(f"🎉 TRAINING SESSION COMPLETE")
print(f"{'='*60}")
print(f"✅ Best Loss: {best_loss:.6f}")
print(f"📊 Total Steps: {total_steps:,}")
print(f"📍 Worker: {ACCOUNT_CONFIG['worker_id']}")
print(f"💾 All progress saved to Drive")
print(f"{'='*60}\n")

# Show final storage info
train_checkpoint.get_storage_info()
```


Output:



```text

============================================================
🎯 TRAINING CONFIGURATION
============================================================
📍 Worker: 2/3
🔄 Epochs: 50
💾 Checkpoint: Every epoch (to Drive)
============================================================

📊 Training Math:
   Dataset: 1,850,519 positions
   Batches/epoch: 7,228
   Steps/epoch: 1,807
   Total steps: 90,350

ℹ️  No checkpoint for Worker 2. Starting fresh.
============================================================
🚀 STARTING TRAINING
============================================================
📍 Worker: 2
🎯 Starting Epoch: 1
🏆 Best Loss: inf
============================================================

Epoch 1/50: 9318it [27:44,  5.79it/s, loss=2.42, lr=0.000407]
```


### Cell 7
```python
# Cell 7: Export Final Model to ONNX
# Compatible: CPU/GPU

import torch
import os

print("📦 Exporting Model to ONNX Format...")

# Load best checkpoint
checkpoint_files = sorted([
    f for f in os.listdir(train_checkpoint.model_checkpoint_dir)
    if f.endswith('.pt')
])

if not checkpoint_files:
    print("❌ No checkpoint found!")
else:
    best_checkpoint_path = os.path.join(
        train_checkpoint.model_checkpoint_dir,
        checkpoint_files[-1]
    )
    
    print(f"📥 Loading best checkpoint: {checkpoint_files[-1]}")
    
    checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Move to CPU for export
    model.to('cpu')
    
    # Dummy input
    dummy_input = torch.randn(1, 12, 8, 8).to('cpu')
    
    # Export path
    onnx_path = os.path.join(PATHS['models'], 'synapse_edge.onnx')
    
    print("⚙️  Exporting to ONNX (this may take 2-3 minutes)...")
    
    try:
        torch.onnx.export(
            model,
            dummy_input,
            onnx_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['board_state'],
            output_names=['policy', 'value', 'tactical'],
            dynamic_axes={
                'board_state': {0: 'batch_size'},
                'policy': {0: 'batch_size'},
                'value': {0: 'batch_size'},
                'tactical': {0: 'batch_size'}
            }
        )
        
        file_size = os.path.getsize(onnx_path) / (1024**2)
        
        print(f"\n{'='*60}")
        print(f"✅ ONNX EXPORT SUCCESSFUL")
        print(f"{'='*60}")
        print(f"📁 Path: {onnx_path}")
        print(f"💾 Size: {file_size:.2f} MB")
        print(f"⚡ Opset Version: 17")
        print(f"{'='*60}\n")
        
        # Optional: Quantize for smaller size
        print("🔧 Optional: Quantizing model for faster inference...")
        print("   (This reduces size from ~60MB to ~15MB)")
        print("   Run: !python -m onnxruntime.quantization.preprocess ...")
        
    except Exception as e:
        print(f"❌ Export failed: {e}")

print("\n✅ Model ready for deployment!")
```


Output:



```text

```

### Cell 8 fine tuning for future
```python
# Cell 8: Fine-tune Synapse-Edge on New Data
# Compatible: GPU ONLY
# Purpose: Fine-tune trained model on master games, self-play, or new puzzles

"""
🎯 HOW TO USE THIS CELL IN FUTURE:

1. Train your initial model with Cells 1-7
2. When you get new data (master games, self-play, etc.):
   - Prepare new database (similar to Cell 7)
   - Update NEW_DATA_PATH below
   - Run this cell
3. Model will fine-tune in 2-3 hours instead of 50+ hours from scratch

BENEFITS:
✅ Keep existing chess knowledge
✅ Fast adaptation to new patterns
✅ Cost-effective training
"""

# ==================== CONFIGURATION ====================

FINETUNE_CONFIG = {
    # Path to your trained model checkpoint
    'base_model_path': '/content/drive/MyDrive/Chessmate_Project/Synapse_Edge/checkpoints/model_checkpoints/synapse_epoch_50_step_50000_finetune.pt',
    
    # New data database (e.g., master games, self-play results)
    'new_data_db': '/content/drive/MyDrive/Chessmate_Project/Synapse_Edge/data/master_games_2024.db',
    
    # Fine-tuning strategy
    'strategy': 'heads_only',  # Options: 'heads_only', 'progressive', 'full'
    
    # Training params
    'epochs': 10,
    'learning_rate': 0.00001,  # Much lower LR for fine-tuning
    'batch_size': 128,
    'gradient_accumulation': 2
}

print(f"{'='*60}")
print(f"🔧 FINE-TUNING CONFIGURATION")
print(f"{'='*60}")
for key, value in FINETUNE_CONFIG.items():
    print(f"   {key}: {value}")
print(f"{'='*60}\n")


# ==================== LOAD BASE MODEL ====================

print("📥 Loading base trained model...")

# Create model with same architecture
finetune_model = create_synapse_edge(
    device=device,
    pretrained_path=FINETUNE_CONFIG['base_model_path'],
    freeze_backbone=False
)

print(f"✅ Base model loaded")
print(f"📊 Model has learned from {training_state['total_steps']:,} training steps\n")


# ==================== APPLY FINE-TUNING STRATEGY ====================

if FINETUNE_CONFIG['strategy'] == 'heads_only':
    print("🔒 Strategy: HEADS ONLY")
    print("   Freezing backbone, training only policy/value/tactical heads")
    finetune_model.freeze_backbone()
    
elif FINETUNE_CONFIG['strategy'] == 'progressive':
    print("🔓 Strategy: PROGRESSIVE UNFREEZING")
    print("   Will gradually unfreeze layers during training")
    finetune_model.progressive_unfreeze(0)
    
elif FINETUNE_CONFIG['strategy'] == 'full':
    print("🔥 Strategy: FULL FINE-TUNING")
    print("   All layers trainable (use with caution - may overfit)")
    finetune_model.unfreeze_backbone()

print(f"🔧 Trainable parameters: {finetune_model.get_trainable_params():,}\n")


# ==================== LOAD NEW DATA ====================

print("📂 Loading new data for fine-tuning...")

from torch.utils.data import DataLoader

finetune_dataset = DistributedChessDataset(
    db_path=FINETUNE_CONFIG['new_data_db'],
    worker_id=0,
    num_workers=1,
    shuffle_buffer=20000
)

finetune_loader = DataLoader(
    finetune_dataset,
    batch_size=FINETUNE_CONFIG['batch_size'],
    num_workers=2,
    pin_memory=True
)

print(f"✅ New data loaded\n")


# ==================== FINE-TUNING OPTIMIZER ====================

# Very low learning rate for fine-tuning
finetune_optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, finetune_model.parameters()),
    lr=FINETUNE_CONFIG['learning_rate'],
    weight_decay=1e-6
)

finetune_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
    finetune_optimizer,
    T_max=FINETUNE_CONFIG['epochs'] * len(finetune_loader),
    eta_min=FINETUNE_CONFIG['learning_rate'] * 0.1
)

finetune_scaler = torch.amp.GradScaler('cuda')


# ==================== FINE-TUNING LOOP ====================

print(f"{'='*60}")
print(f"🚀 STARTING FINE-TUNING")
print(f"{'='*60}\n")

finetune_best_loss = float('inf')

for epoch in range(1, FINETUNE_CONFIG['epochs'] + 1):
    
    finetune_model.train()
    epoch_loss = 0.0
    
    pbar = tqdm(finetune_loader, desc=f"Fine-tune Epoch {epoch}/{FINETUNE_CONFIG['epochs']}")
    
    for batch_idx, (board, policy_target, value_target, tactical_target) in enumerate(pbar):
        
        board = board.to(device)
        policy_target = policy_target.to(device)
        value_target = value_target.to(device)
        tactical_target = tactical_target.to(device)
        
        with torch.amp.autocast('cuda'):
            policy_pred, value_pred, tactical_pred = finetune_model(board)
            
            loss, loss_dict = criterion(
                (policy_pred, value_pred, tactical_pred),
                (policy_target, value_target, tactical_target),
                finetune_model
            )
            
            loss = loss / FINETUNE_CONFIG['gradient_accumulation']
        
        finetune_scaler.scale(loss).backward()
        
        if (batch_idx + 1) % FINETUNE_CONFIG['gradient_accumulation'] == 0:
            finetune_scaler.step(finetune_optimizer)
            finetune_scaler.update()
            finetune_optimizer.zero_grad()
            finetune_scheduler.step()
        
        epoch_loss += loss_dict['total']
        pbar.set_postfix({'loss': loss_dict['total']})
    
    avg_loss = epoch_loss / len(finetune_loader)
    
    # Save best fine-tuned model
    if avg_loss < finetune_best_loss:
        finetune_best_loss = avg_loss
        
        finetuned_path = os.path.join(
            PATHS['models'],
            f'synapse_finetuned_epoch_{epoch}.pt'
        )
        
        finetune_model.save_for_finetuning(finetuned_path)
        print(f"\n💾 Best fine-tuned model saved: {finetuned_path}")
        print(f"   Loss: {finetune_best_loss:.6f}\n")
    
    print(f"Epoch {epoch} | Avg Loss: {avg_loss:.6f}\n")
    
    # Progressive unfreezing (if strategy is progressive)
    if FINETUNE_CONFIG['strategy'] == 'progressive' and epoch % 2 == 0:
        current_stage = min(epoch // 2, 4)
        finetune_model.progressive_unfreeze(current_stage)
        print(f"🔓 Unfroze to stage {current_stage}\n")

print(f"{'='*60}")
print(f"✅ FINE-TUNING COMPLETE")
print(f"{'='*60}")
print(f"🏆 Best Fine-tuned Loss: {finetune_best_loss:.6f}")
print(f"📦 Final Model: {finetuned_path}")
print(f"{'='*60}\n")

print("\n📝 NOTES:")
print("   1. This model now incorporates new patterns from your additional data")
print("   2. Original chess knowledge from base model is preserved")
print("   3. You can repeat this process with more new data anytime")
print("   4. Export to ONNX using Cell 7 with this fine-tuned checkpoint")
```


Output:



```text

```
