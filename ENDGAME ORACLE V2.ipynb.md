## Notebook Name: ENDGAME ORACLE V2.ipynb.md

---

## 1. Introduction
eta amar  endgame er  notebook. 

---

### Cell 1: Environment Setup
```python




# Cell 1: Environment Setup & Worker Configuration
# ==============================================================================
# 🔮 ENDGAME ORACLE V2 - DISTRIBUTED WORKER INIT
# ==============================================================================

import os
import time
import threading
import random
from google.colab import drive

# ==============================================================================
# ⚠️ WORKER CONFIGURATION - CHANGE THIS FOR EACH ACCOUNT!
# Account 1 → WORKER_ID = 1 (3-piece endgames)
# Account 2 → WORKER_ID = 2 (4-piece endgames)
# Account 3 → WORKER_ID = 3 (5-piece endgames)
# ==============================================================================
WORKER_ID = 1  # ⚠️ CHANGE THIS VALUE: 1, 2, or 3
# ==============================================================================

print(f"⚙️ Setting up Worker Node #{WORKER_ID}...")

# Install dependencies
!pip install python-chess requests

import chess
import chess.syzygy

# Mount Drive
print("\n🔗 Connecting to Google Drive...")
drive.mount('/content/drive')

# Shared Workspace
PROJECT_ROOT = '/content/drive/MyDrive/GambitFlow_Project'
DATA_FACTORY_DIR = os.path.join(PROJECT_ROOT, 'Synapse_Data')
os.makedirs(DATA_FACTORY_DIR, exist_ok=True)

# Set unique random seed per worker
random.seed(WORKER_ID * 1000)
print(f"✅ Worker {WORKER_ID} Initialized (Seed: {WORKER_ID * 1000})")

# Keep-Alive
def keep_colab_awake():
    while True:
        time.sleep(60)

threading.Thread(target=keep_colab_awake, daemon=True).start()
print("✅ Keep-Alive Active.")

        

```


Output:



```text

⚙️ Setting up Worker Node #1...
🔗 Connecting to Google Drive...
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
✅ Worker 1 Initialized (Seed: 1000)
✅ Keep-Alive Active.


```

---

### Cell 2
```python



# Cell 2: Download Syzygy Tablebases (3-4-5 piece WDL)
# ==============================================================================
# 📥 DOWNLOAD PERFECT ENDGAME KNOWLEDGE
# ==============================================================================

import os

SYZYGY_DIR = '/content/syzygy'
os.makedirs(SYZYGY_DIR, exist_ok=True)

BASE_URL = "https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/"

# File mapping: (server_name -> local_name)
FILES_TO_DOWNLOAD = {
    # 3-Piece
    "KQvK.rtbw": "KQvK.rtbw",
    "KRvK.rtbw": "KRvK.rtbw",
    "KBvK.rtbw": "KBvK.rtbw",
    "KNvK.rtbw": "KNvK.rtbw",
    "KPvK.rtbw": "KPvK.rtbw",

    # 4-Piece (Queen)
    "KQvKR.rtbw": "KQvKR.rtbw",
    "KQvKB.rtbw": "KQvKB.rtbw",
    "KQvKN.rtbw": "KQvKN.rtbw",
    "KQvKP.rtbw": "KQvKP.rtbw",
    "KQQvK.rtbw": "KQQvK.rtbw",

    # 4-Piece (Rook)
    "KRvKR.rtbw": "KRvKR.rtbw",
    "KRvKB.rtbw": "KRvKB.rtbw",
    "KRvKN.rtbw": "KRvKN.rtbw",
    "KRvKP.rtbw": "KRvKP.rtbw",
    "KRRvK.rtbw": "KRRvK.rtbw",

    # 4-Piece (Minor)
    "KBvKB.rtbw": "KBvKB.rtbw",
    "KBvKN.rtbw": "KBvKN.rtbw",
    "KBvKP.rtbw": "KBvKP.rtbw",
    "KNvKN.rtbw": "KNvKN.rtbw",
    "KNvKP.rtbw": "KNvKP.rtbw",

    # 4-Piece (Pawn)
    "KPvKP.rtbw": "KPvKP.rtbw",
    "KPPvK.rtbw": "KPPvK.rtbw",

    # 5-Piece (Sample - add more if needed)
    "KQRvK.rtbw": "KQRvK.rtbw",
    "KQBvK.rtbw": "KQBvK.rtbw",
    "KQNvK.rtbw": "KQNvK.rtbw",
    "KRRvKR.rtbw": "KRRvKR.rtbw",
    "KRBvK.rtbw": "KRBvK.rtbw",
    "KRNvK.rtbw": "KRNvK.rtbw",
    "KBBvK.rtbw": "KBBvK.rtbw",
    "KBNvK.rtbw": "KBNvK.rtbw",
    "KNNvK.rtbw": "KNNvK.rtbw",
}

print(f"🚀 Starting Download from: {BASE_URL}")
print(f"📂 Saving to: {SYZYGY_DIR}\n")

success_count = 0

for server_name, local_name in FILES_TO_DOWNLOAD.items():
    url = BASE_URL + server_name
    save_path = os.path.join(SYZYGY_DIR, local_name)

    # Skip if exists
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"✅ Exists: {local_name}")
        success_count += 1
        continue

    # Download with wget
    cmd = f'wget -q --show-progress -O {save_path} {url}'
    exit_code = os.system(cmd)

    if exit_code == 0 and os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        print(f"✅ Downloaded: {local_name}")
        success_count += 1
    else:
        print(f"❌ Failed: {local_name}")
        if os.path.exists(save_path):
            os.remove(save_path)

print("-" * 50)
print(f"📊 Download Complete: {success_count}/{len(FILES_TO_DOWNLOAD)}")

if success_count > 15:
    print("✅ Sufficient tablebases available. Ready for next cell.")
else:
    print("⚠️ Many downloads failed. Check network connection.")


     
            



```


Output:



```text
🚀 Starting Download from: https://tablebase.lichess.ovh/tables/standard/3-4-5-wdl/
📂 Saving to: /content/syzygy
--------------------------------------------------
📊 Download Complete: 31/31
✅ Sufficient tablebases available. Ready for next cell.


```

---

### Cell 3
```python




# Cell 3: Distributed Endgame Position Generation
# ==============================================================================
# 🔮 GENERATE ENDGAME POSITIONS BY WORKER SPECIALIZATION
# ==============================================================================

import sqlite3
import chess
import chess.syzygy
import random
import os
import shutil
from tqdm import tqdm

# Configuration
LOCAL_DB_NAME = f"endgame_oracle_w{WORKER_ID}.db"
DRIVE_DB_PATH = os.path.join(DATA_FACTORY_DIR, LOCAL_DB_NAME)
TARGET_POSITIONS = 400000

# Worker Specialization
# Worker 1: 3-piece endgames
# Worker 2: 4-piece endgames
# Worker 3: 5-piece endgames

if WORKER_ID == 1:
    PIECE_COUNT = 3
    SCENARIOS = [
        (['Q'], []), (['R'], []), (['B'], []), (['N'], []), (['P'], [])
    ]
elif WORKER_ID == 2:
    PIECE_COUNT = 4
    SCENARIOS = [
        (['Q'], ['R']), (['Q'], ['B']), (['Q'], ['N']), (['Q'], ['P']),
        (['R'], ['R']), (['R'], ['B']), (['R'], ['N']), (['R'], ['P']),
        (['B'], ['B']), (['B'], ['N']), (['B'], ['P']),
        (['N'], ['N']), (['N'], ['P']), (['P'], ['P']),
        (['Q', 'Q'], []), (['R', 'R'], []), (['P', 'P'], [])
    ]
elif WORKER_ID == 3:
    PIECE_COUNT = 5
    SCENARIOS = [
        (['Q', 'R'], []), (['Q', 'B'], []), (['Q', 'N'], []),
        (['R', 'R'], ['R']), (['R', 'B'], []), (['R', 'N'], []),
        (['B', 'B'], []), (['B', 'N'], []), (['N', 'N'], []),
        (['Q'], ['R', 'R']), (['Q'], ['R', 'B']),
        (['R'], ['R', 'R']), (['R'], ['B', 'B'])
    ]
else:
    print("❌ Invalid WORKER_ID. Must be 1, 2, or 3.")
    raise ValueError("WORKER_ID must be 1, 2, or 3")

print(f"🎯 Worker {WORKER_ID}: Generating {PIECE_COUNT}-piece endgames")
print(f"📊 Scenarios: {len(SCENARIOS)}")

# Initialize Tablebase
print("🔧 Loading Syzygy Tablebases...")
try:
    tablebase = chess.syzygy.open_tablebase(SYZYGY_DIR)
    print("✅ Tablebase loaded successfully")
except Exception as e:
    print(f"❌ Failed to load tablebase: {e}")
    raise

# Test tablebase
test_board = chess.Board("4k3/8/8/8/8/8/3Q4/4K3 w - - 0 1")
try:
    wdl = tablebase.probe_wdl(test_board)
    print(f"✅ Tablebase test passed (KQK WDL: {wdl})")
except Exception as e:
    print(f"❌ Tablebase test failed: {e}")
    raise

# Database Setup
conn = sqlite3.connect(LOCAL_DB_NAME)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS endgame (
        fen TEXT PRIMARY KEY,
        wdl INTEGER,
        piece_count INTEGER
    )
''')
cursor.execute('PRAGMA synchronous = OFF')
cursor.execute('PRAGMA journal_mode = MEMORY')

# Position Generator
def generate_endgame_position():
    """Generate random endgame position"""
    board = chess.Board(None)

    # Place kings (must be legal distance)
    while True:
        k1, k2 = random.sample(range(64), 2)
        if chess.square_distance(k1, k2) > 1:
            board.set_piece_at(k1, chess.Piece(chess.KING, chess.WHITE))
            board.set_piece_at(k2, chess.Piece(chess.KING, chess.BLACK))
            break

    # Select random scenario
    white_extras, black_extras = random.choice(SCENARIOS)

    # Place white pieces
    for p_char in white_extras:
        pt = chess.PIECE_SYMBOLS.index(p_char.lower())
        placed = False
        attempts = 0

        while not placed and attempts < 50:
            sq = random.randint(0, 63)
            if board.piece_at(sq) is None:
                # No pawns on back ranks
                if pt == chess.PAWN and chess.square_rank(sq) in [0, 7]:
                    attempts += 1
                    continue

                board.set_piece_at(sq, chess.Piece(pt, chess.WHITE))
                placed = True
            attempts += 1

    # Place black pieces
    for p_char in black_extras:
        pt = chess.PIECE_SYMBOLS.index(p_char.lower())
        placed = False
        attempts = 0

        while not placed and attempts < 50:
            sq = random.randint(0, 63)
            if board.piece_at(sq) is None:
                # No pawns on back ranks
                if pt == chess.PAWN and chess.square_rank(sq) in [0, 7]:
                    attempts += 1
                    continue

                board.set_piece_at(sq, chess.Piece(pt, chess.BLACK))
                placed = True
            attempts += 1

    # Random turn
    board.turn = random.choice([chess.WHITE, chess.BLACK])

    return board

# Check existing progress
try:
    cursor.execute("SELECT COUNT(*) FROM endgame")
    gen_count = cursor.fetchone()[0]
except:
    gen_count = 0

print(f"\n🚀 Worker {WORKER_ID}: Starting generation...")
print(f"📈 Current progress: {gen_count:,}/{TARGET_POSITIONS:,}")

# Main Generation Loop
batch_data = []
pbar = tqdm(total=TARGET_POSITIONS, initial=gen_count, unit="pos", colour='green')

try:
    while gen_count < TARGET_POSITIONS:
        board = generate_endgame_position()

        # Validate position
        if not board.is_valid():
            continue

        if board.is_checkmate() or board.is_stalemate():
            continue

        # Probe tablebase
        try:
            wdl = tablebase.probe_wdl(board)

            # Normalize WDL to -1, 0, 1
            wdl_normalized = 1 if wdl > 0 else (-1 if wdl < 0 else 0)

            # Create FEN (without move counters)
            fen = f"{board.board_fen()} {'w' if board.turn else 'b'} - -"

            batch_data.append((fen, wdl_normalized, PIECE_COUNT))
            gen_count += 1
            pbar.update(1)

            # Batch insert every 2000 positions
            if len(batch_data) >= 2000:
                cursor.executemany(
                    'INSERT OR IGNORE INTO endgame VALUES (?, ?, ?)',
                    batch_data
                )
                conn.commit()
                batch_data = []

                # Periodic backup to Drive (every 10K positions)
                if gen_count % 10000 == 0:
                    shutil.copy(LOCAL_DB_NAME, DRIVE_DB_PATH)

        except KeyboardInterrupt:
            print("\n⏸️ Paused by user")
            break
        except:
            # Skip positions not in tablebase
            continue

except KeyboardInterrupt:
    print("\n🛑 Generation paused")

finally:
    # Save remaining batch
    if batch_data:
        cursor.executemany(
            'INSERT OR IGNORE INTO endgame VALUES (?, ?, ?)',
            batch_data
        )
        conn.commit()

    conn.close()
    pbar.close()

    # Final backup to Drive
    shutil.copy(LOCAL_DB_NAME, DRIVE_DB_PATH)

    print(f"\n✅ Session saved")
    print(f"📊 Total positions: {gen_count:,}/{TARGET_POSITIONS:,}")
    print(f"💾 Database: {DRIVE_DB_PATH}")
 
   

```


Output:



```text

🎯 Worker 1: Generating 3-piece endgames
📊 Scenarios: 5
🔧 Loading Syzygy Tablebases...
✅ Tablebase loaded successfully
✅ Tablebase test passed (KQK WDL: 2)

🚀 Worker 1: Starting generation...
📈 Current progress: 190,061/400,000
100%|██████████| 400000/400000 [00:35<00:00, 5843.85pos/s]

✅ Session saved
📊 Total positions: 400,000/400,000
💾 Database: /content/drive/MyDrive/GambitFlow_Project/Synapse_Data/endgame_oracle_w1.db


  
```

---

### Cell 4
```python



# ==============================================================================
# 🚀 UPLOAD ENDGAME DATABASES TO HUGGING FACE
# ==============================================================================

from huggingface_hub import HfApi
import os

# --- 1. Configuration ---
# ⚠️ আপনার Hugging Face Write Token টি এখানে দিন
HF_TOKEN = "HF"

HF_USERNAME = "GambitFlow"
REPO_ID = f"{HF_USERNAME}/Endgame-Tablebase"
DATA_DIR = "/content/drive/MyDrive/GambitFlow_Project/Synapse_Data"

# Files to upload
FILES_TO_UPLOAD = {
    "endgame_oracle_w1.db": "endgame/3_piece.db", # Worker 1 -> 3-piece
    "endgame_oracle_w2.db": "endgame/4_piece.db", # Worker 2 -> 4-piece
    "endgame_oracle_w3.db": "endgame/5_piece.db", # Worker 3 -> 5-piece
}

print(f"🚀 Initializing upload to: {REPO_ID}")
print("-" * 50)

# --- 2. Upload Logic ---
try:
    api = HfApi(token=HF_TOKEN)

    # Create the repo if it doesn't exist
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    for local_name, repo_path in FILES_TO_UPLOAD.items():
        source_path = os.path.join(DATA_DIR, local_name)

        if not os.path.exists(source_path):
            print(f"⚠️ SKIPPING: File not found at {source_path}")
            continue

        file_size_mb = os.path.getsize(source_path) / (1024**2)
        print(f"\n⏳ Uploading {local_name} ({file_size_mb:.2f} MB) to '{repo_path}'...")

        # Upload the file
        api.upload_file(
            path_or_fileobj=source_path,
            path_in_repo=repo_path,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Add {os.path.basename(repo_path)} endgame data"
        )
        print(f"   -> ✅ Upload successful!")

    print("\n" + "=" * 50)
    print("🎉 All endgame databases have been uploaded successfully!")
    print(f"🔗 Dataset URL: https://huggingface.co/datasets/{REPO_ID}")

except Exception as e:
    print(f"\n❌ An error occurred during upload: {e}")
    print("👉 Tip: Please check if your HF_TOKEN is correct and has 'Write' permissions.")

 
   

```


Output:



```text


 🚀 Initializing upload to: GambitFlow/Endgame-Tablebase
--------------------------------------------------

⏳ Uploading endgame_oracle_w1.db (27.09 MB) to 'endgame/3_piece.db'...
Processing Files (1 / 1)      : 100% 28.4MB / 28.4MB, 8.24MB/s  New Data Upload               :   0.00B /  0.00B,  0.00B/s    ...Data/endgame_oracle_w1.db: 100% 28.4MB / 28.4MB               -> ✅ Upload successful!

⏳ Uploading endgame_oracle_w2.db (15.36 MB) to 'endgame/4_piece.db'...
Processing Files (1 / 1)      : 100% 16.1MB / 16.1MB,  0.00B/s  New Data Upload               :   0.00B /  0.00B,  0.00B/s    ...Data/endgame_oracle_w2.db: 100% 16.1MB / 16.1MB               -> ✅ Upload successful!

⏳ Uploading endgame_oracle_w3.db (31.14 MB) to 'endgame/5_piece.db'...
Processing Files (1 / 1)      : 100% 32.6MB / 32.6MB, 26.5MB/s  New Data Upload               :   0.00B /  0.00B,  0.00B/s    ...Data/endgame_oracle_w3.db: 100% 32.6MB / 32.6MB               -> ✅ Upload successful!

==================================================
🎉 All endgame databases have been uploaded successfully!
🔗 Dataset URL: https://huggingface.co/datasets/GambitFlow/Endgame-Tablebase

  
```
