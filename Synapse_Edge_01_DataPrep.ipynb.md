## Notebook Name: Synapse_Edge_01_DataPrep.ipynb

---

## 1. Introduction
eta amar  GambitFlow/Synapse-Edge  model er dataset bananor notebook. 

---

### Cell 1
```python

# Cell 1: Environment Setup & Keep Alive System
# Compatible: CPU/GPU

import os
import json
import time
import threading
from google.colab import drive
from datetime import datetime

# ১. Drive Mount
drive.mount('/content/drive')

# ২. Project Structure Setup
BASE_PATH = '/content/drive/MyDrive/Chessmate_Project/Synapse_Edge'
PATHS = {
    'checkpoints': os.path.join(BASE_PATH, 'checkpoints'),
    'data': os.path.join(BASE_PATH, 'data'),
    'models': os.path.join(BASE_PATH, 'models'),
    'logs': os.path.join(BASE_PATH, 'logs')
}

for path in PATHS.values():
    os.makedirs(path, exist_ok=True)

print(f"✅ Project Directory: {BASE_PATH}")

# ৩. Keep Alive System (Prevents Colab Disconnect)
class KeepAlive:
    def __init__(self, interval=60):
        self.interval = interval
        self.running = True
        self.thread = threading.Thread(target=self._keep_alive, daemon=True)

    def start(self):
        self.thread.start()
        print("🔄 Keep-Alive System Started")

    def _keep_alive(self):
        while self.running:
            print(f"💓 [{datetime.now().strftime('%H:%M:%S')}] Keep-Alive Signal")
            time.sleep(self.interval)

    def stop(self):
        self.running = False

# Start Keep-Alive
keeper = KeepAlive(interval=120)  # Every 2 minutes
keeper.start()

# ৪. Checkpoint System
class CheckpointManager:
    def __init__(self, checkpoint_dir):
        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(checkpoint_dir, 'progress.json')

    def load(self):
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, 'r') as f:
                data = json.load(f)
            print(f"📂 Checkpoint Loaded: {data}")
            return data
        return {
            "stage": "data_prep",
            "games_processed": 0,
            "unique_positions": 0,
            "puzzles_processed": 0,
            "last_updated": None
        }

    def save(self, data):
        data['last_updated'] = datetime.now().isoformat()
        with open(self.checkpoint_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"💾 Checkpoint Saved: {data['stage']}")

checkpoint_mgr = CheckpointManager(PATHS['checkpoints'])
state = checkpoint_mgr.load()

print("\n" + "="*60)
print("🚀 SYNAPSE-EDGE: DATA PREPARATION READY")
print("="*60)
```


Output:



```text
Mounted at /content/drive
✅ Project Directory: /content/drive/MyDrive/Chessmate_Project/Synapse_Edge
💓 [11:01:01] Keep-Alive Signal
🔄 Keep-Alive System Started
📂 Checkpoint Loaded: {'stage': 'data_ready', 'games_processed': 0, 'unique_positions': 2554920, 'puzzles_processed': 5600086, 'last_updated': '2025-12-25T15:29:30.970446', 'elite_db_path': '/content/drive/MyDrive/Chessmate_Project/Synapse_Edge/data/chess_stats_v2.db', 'puzzle_db_path': '/content/drive/MyDrive/Chessmate_Project/Synapse_Edge/data/synapse_puzzles.db', 'tactical_positions': 3007724, 'final_db_path': '/content/drive/MyDrive/Chessmate_Project/Synapse_Edge/data/synapse_training_final.db', 'total_samples': 5551558}

============================================================
🚀 SYNAPSE-EDGE: DATA PREPARATION READY
============================================================
```

---

### Cell 2
```python
# Cell 2: Install All Dependencies
# Compatible: CPU/GPU

print("📦 Installing Dependencies...")

# Core Libraries
!pip install -q python-chess==1.999
!pip install -q zstandard
!pip install -q requests
!pip install -q tqdm

# Verify Installations
import chess.pgn
import zstandard as zstd
import requests
from tqdm import tqdm

print("✅ All Dependencies Installed Successfully")

# Display System Info
import sys
print(f"\n🖥️  Python Version: {sys.version}")
print(f"📍 Working Directory: {os.getcwd()}")
 

```


Output:



```text
📦 Installing Dependencies...
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.1/6.1 MB 67.3 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
  Building wheel for chess (setup.py) ... done
✅ All Dependencies Installed Successfully

🖥️  Python Version: 3.12.12 (main, Oct 10 2025, 08:52:57) [GCC 11.4.0]
📍 Working Directory: /content
```

---

### Cell 3
```python


       # Cell 3: Load Existing Elite Dataset from Hugging Face
# Compatible: CPU/GPU
# Purpose: Download your existing 5M+ positions database

from huggingface_hub import hf_hub_download
import os
import shutil

print("📥 Loading Existing Elite Dataset from Hugging Face...")
print("🔗 Repository: GambitFlow/Elite-Data")

# Download database from your HF repo
elite_db_path = hf_hub_download(
    repo_id="GambitFlow/Elite-Data",
    filename="chess_stats_v2.db",
    repo_type="dataset",
    local_dir=PATHS['data']
)

print(f"✅ Dataset Downloaded: {elite_db_path}")
print(f"💾 Size: {os.path.getsize(elite_db_path) / (1024**3):.2f} GB")

# Verify database
import sqlite3
conn = sqlite3.connect(elite_db_path)
cursor = conn.cursor()

cursor.execute("SELECT COUNT(*) FROM positions")
total_positions = cursor.fetchone()[0]

cursor.execute("SELECT fen, stats FROM positions LIMIT 1")
sample = cursor.fetchone()
conn.close()

print(f"\n📊 Database Statistics:")
print(f"   - Total Positions: {total_positions:,}")
print(f"   - Sample FEN: {sample[0][:50]}...")

# Save to checkpoint
state['elite_db_path'] = elite_db_path
state['unique_positions'] = total_positions
state['stage'] = 'elite_data_loaded'
checkpoint_mgr.save(state)

print("\n✅ Existing dataset loaded successfully!")
print("⏩ You can skip to Cell 4 (Puzzles) now.")            
                    


```


Output:



```text
📥 Loading Existing Elite Dataset from Hugging Face...
🔗 Repository: GambitFlow/Elite-Data
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
chess_stats_v2.db: 100%
 925M/925M [00:08<00:00, 190MB/s]
✅ Dataset Downloaded: /content/drive/MyDrive/Chessmate_Project/Synapse_Edge/data/chess_stats_v2.db
💾 Size: 0.86 GB

📊 Database Statistics:
   - Total Positions: 2,554,920
   - Sample FEN: rnbq1rk1/pp1p1ppp/4p3/2P5/2P1n3/P1Q5/1P2PPPP/R1B1K...
💾 Checkpoint Saved: elite_data_loaded

✅ Existing dataset loaded successfully!
⏩ You can skip to Cell 4 (Puzzles) now.

 

  
```

---

### Cell 4
```python
# Cell 4: Download Tactical Puzzles Database
# Compatible: CPU/GPU
# Logic: Defined robust streaming download with progress tracking

import requests
import os
from tqdm import tqdm

def download_database(url, destination):
    """
    Downloads large files using streaming to avoid memory overflow.
    Includes a professional progress bar.
    """
    print(f"📡 Requesting: {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get('content-length', 0))
    block_size = 1024 * 1024  # 1 MB

    if response.status_code == 200:
        with open(destination, 'wb') as f:
            with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                for data in response.iter_content(block_size):
                    f.write(data)
                    pbar.update(len(data))
        print(f"\n✅ File successfully saved to: {destination}")
    else:
        print(f"❌ Error: Failed to download. Status Code: {response.status_code}")

# Lichess Puzzles Database (3M+ puzzles)
PUZZLE_URL = 'https://database.lichess.org/lichess_db_puzzle.csv.zst'
puzzle_path = os.path.join(PATHS['data'], 'lichess_puzzles.csv.zst')

print("🧩 Initializing Tactical Puzzles Download...")
print("📊 Estimated Size: ~500 MB | Target: 3,000,000+ tactical patterns")
print("🎯 Strategy: Teaching Synapse-Edge to recognize forks, pins, and checkmate patterns.")

# Execute Download
try:
    download_database(PUZZLE_URL, puzzle_path)
except Exception as e:
    print(f"⚠️ Critical Download Error: {e}")

# Save state to checkpoint
state['puzzle_db_path'] = puzzle_path
state['stage'] = 'puzzles_downloaded'
checkpoint_mgr.save(state)

print("\n" + "="*60)
print("✅ TACTICAL DATABASE DOWNLOAD COMPLETE")
print("="*60)
# Note: db_path variable from previous cells is used for size reporting
# If cell 3 was successful, it's already in state
print(f"📂 Elite Dataset Path: {state.get('elite_db_path', 'Not found')}")
print(f"🧩 Puzzles Path: {puzzle_path}")
```


Output:



```text
🧩 Initializing Tactical Puzzles Download...
📊 Estimated Size: ~500 MB | Target: 3,000,000+ tactical patterns
🎯 Strategy: Teaching Synapse-Edge to recognize forks, pins, and checkmate patterns.
📡 Requesting: https://database.lichess.org/lichess_db_puzzle.csv.zst
Downloading: 100%|██████████| 279M/279M [00:19<00:00, 14.6MB/s]
✅ File successfully saved to: /content/drive/MyDrive/Chessmate_Project/Synapse_Edge/data/lichess_puzzles.csv.zst
💾 Checkpoint Saved: puzzles_downloaded

============================================================
✅ TACTICAL DATABASE DOWNLOAD COMPLETE
============================================================
📂 Elite Dataset Path: /content/drive/MyDrive/Chessmate_Project/Synapse_Edge/data/chess_stats_v2.db
🧩 Puzzles Path: /content/drive/MyDrive/Chessmate_Project/Synapse_Edge/data/lichess_puzzles.csv.zst


```


### Cell 5

```text
skipped

```


### Cell 6
```python
# Cell 6: Process Tactical Puzzles with Local Buffering
# Logic: Copy to local VM first to avoid Drive I/O bottlenecks

import csv
import zstandard as zstd
import io
import sqlite3
import os
import shutil
from tqdm import tqdm

def process_puzzles():
    print("🚀 Starting High-Speed Puzzle Processing...")

    # ১. ফাইল পাথ সেটআপ
    drive_zst_path = os.path.join(PATHS['data'], 'lichess_puzzles.csv.zst')
    local_zst_path = '/content/puzzles.csv.zst' # কোল্যাব লোকাল স্টোরেজ
    local_db_path = '/content/synapse_puzzles.db'
    final_db_path = os.path.join(PATHS['data'], 'synapse_puzzles.db')

    # ২. ড্রাইভ থেকে লোকালে কপি করা (এটি অত্যন্ত জরুরি)
    if not os.path.exists(local_zst_path):
        print("📥 Copying file to local VM for high-speed access...")
        shutil.copy(drive_zst_path, local_zst_path)
        print("✅ Copy complete.")

    # ৩. SQLite লোকাল ডাটাবেস তৈরি
    conn = sqlite3.connect(local_db_path)
    cursor = conn.cursor()
    cursor.execute('DROP TABLE IF EXISTS puzzles')
    cursor.execute('''
        CREATE TABLE puzzles (
            fen TEXT PRIMARY KEY,
            best_move TEXT,
            themes TEXT,
            rating INTEGER,
            difficulty TEXT
        )
    ''')
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('PRAGMA journal_mode = MEMORY')

    # ৪. ফিল্টার এবং প্রসেসিং
    target_themes = ['fork', 'pin', 'skewer', 'discoveredAttack', 'mate', 'mateIn2', 'mateIn3', 'checkmate']
    processed = 0
    tactical_positions = 0
    batch = []

    print("🧩 Decompressing and Filtering Tactics...")

    try:
        dctx = zstd.ZstdDecompressor()
        with open(local_zst_path, 'rb') as ifh:
            with dctx.stream_reader(ifh) as reader:
                text_stream = io.TextIOWrapper(reader, encoding='utf-8')
                csv_reader = csv.reader(text_stream)

                # হেডার পড়া
                header = next(csv_reader)

                # প্রোগ্রেস বার (৩.৫ মিলিয়ন পাজল এভারেজ)
                with tqdm(total=3500000, desc="Extracting", unit=" puzzles") as pbar:
                    for row in csv_reader:
                        try:
                            # Row structure: [PuzzleId, FEN, Moves, Rating, ..., Themes, ...]
                            if len(row) < 8: continue

                            fen = row[1]
                            moves = row[2]
                            rating = int(row[3])
                            themes = row[7]

                            processed += 1

                            # ট্যাকটিক্যাল থিম ফিল্টার
                            if any(t in themes for t in target_themes):
                                first_move = moves.split()[0]
                                clean_fen = " ".join(fen.split(" ")[:4])

                                # ডিফিকাল্টি লেভেল
                                if rating < 1500: diff = "beginner"
                                elif rating < 2000: diff = "intermediate"
                                else: diff = "advanced"

                                batch.append((clean_fen, first_move, themes, rating, diff))
                                tactical_positions += 1

                            # প্রতি ৫০০০ পাজল পর পর সেভ
                            if len(batch) >= 5000:
                                cursor.executemany('INSERT OR IGNORE INTO puzzles VALUES (?,?,?,?,?)', batch)
                                conn.commit()
                                batch = []
                                pbar.update(processed - pbar.n)

                        except Exception:
                            continue

                    # ফাইনাল ব্যাচ
                    if batch:
                        cursor.executemany('INSERT OR IGNORE INTO puzzles VALUES (?,?,?,?,?)', batch)
                        conn.commit()
                        pbar.update(processed - pbar.n)

    except Exception as e:
        print(f"❌ Error: {e}")
    finally:
        conn.close()

    # ৫. লোকাল ডাটাবেস ড্রাইভে মুভ করা
    if tactical_positions > 0:
        print("\n💾 Moving processed database to Google Drive...")
        shutil.move(local_db_path, final_db_path)

        # সেশন স্টেট আপডেট
        state['puzzles_processed'] = processed
        state['tactical_positions'] = tactical_positions
        state['puzzle_db_path'] = final_db_path
        state['stage'] = 'puzzles_processed'
        checkpoint_mgr.save(state)

        print(f"\n{'='*60}")
        print(f"✅ SUCCESS: {tactical_positions:,} TACTICS READY")
        print(f"{'='*60}")
        print(f"📊 Total Scanned: {processed:,}")
        print(f"💾 Saved to: {final_db_path}")
    else:
        print("❌ Fatal Error: No data processed. Check file integrity.")

# রান করা
process_puzzles()
```


Output:



```text

🚀 Starting High-Speed Puzzle Processing...
📥 Copying file to local VM for high-speed access...
✅ Copy complete.
🧩 Decompressing and Filtering Tactics...
Extracting:  97%|█████████▋| 3408802/3500000 [00:57<00:01, 56016.65 puzzles/s]💓 [11:03:01] Keep-Alive Signal
Extracting: 5600086 puzzles [01:54, 49077.32 puzzles/s]

💾 Moving processed database to Google Drive...
💾 Checkpoint Saved: puzzles_processed

============================================================
✅ SUCCESS: 3,007,724 TACTICS READY
============================================================
📊 Total Scanned: 5,600,086
💾 Saved to: /content/drive/MyDrive/Chessmate_Project/Synapse_Edge/data/synapse_puzzles.db
```


### Cell 7
```python
# Cell 7: Final Database Merge with Auto-Validation
# Logic: High-speed SQL merging with path and table validation

import sqlite3
import json
import os
import shutil
from tqdm import tqdm

def merge_databases():
    print("🔀 Starting Synapse-Edge Final Merge...")

    # ১. ফাইল পাথ ভ্যালিডেশন
    # আমরা সরাসরি চেক করব ফাইলগুলো ড্রাইভে আছে কিনা
    elite_db_drive = os.path.join(PATHS['data'], 'chess_stats_v2.db')
    puzzle_db_drive = os.path.join(PATHS['data'], 'synapse_puzzles.db')
    final_drive_db = os.path.join(PATHS['data'], 'synapse_training_final.db')

    if not os.path.exists(elite_db_drive):
        print(f"❌ Error: Elite Database not found at {elite_db_drive}")
        return
    if not os.path.exists(puzzle_db_drive):
        print(f"❌ Error: Puzzle Database not found at {puzzle_db_drive}")
        return

    # ২. স্পিড বাড়ানোর জন্য লোকালে কপি
    print("📥 Moving files to local VM storage for 10x speed...")
    local_elite = '/content/elite_v2.db'
    local_puzzle = '/content/puzzle_v3.db'
    local_final = '/content/synapse_final.db'

    shutil.copy(elite_db_drive, local_elite)
    shutil.copy(puzzle_db_drive, local_puzzle)

    # ৩. ডাটাবেস কানেকশন এবং টেবিল চেক
    print("🔍 Validating source tables...")
    def check_table(db_path, table_name):
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table_name,))
        exists = c.fetchone()
        conn.close()
        return exists

    if not check_table(local_elite, 'positions'):
        print(f"❌ Error: Table 'positions' not found in Elite DB. Please check Cell 3.")
        return
    if not check_table(local_puzzle, 'puzzles'):
        print(f"❌ Error: Table 'puzzles' not found in Puzzle DB. Please check Cell 6.")
        return

    # ৪. মার্জিং প্রসেস শুরু
    final_conn = sqlite3.connect(local_final)
    final_cursor = final_conn.cursor()

    # নতুন টেবিল তৈরি (V3 Schema)
    final_cursor.execute('DROP TABLE IF EXISTS training_data')
    final_cursor.execute('''
        CREATE TABLE training_data (
            fen TEXT PRIMARY KEY,
            position_stats TEXT,
            best_move TEXT,
            is_tactical INTEGER,
            difficulty TEXT
        )
    ''')

    final_cursor.execute('PRAGMA synchronous = OFF')
    final_cursor.execute('PRAGMA journal_mode = MEMORY')

    # ৫. ইমপোর্ট Elite Data (SQL-to-SQL Attach - Fastest Method)
    print("\n♟️  Importing 5M Elite Data...")
    final_cursor.execute(f"ATTACH DATABASE '{local_elite}' AS elite_db")
    final_cursor.execute('''
        INSERT OR IGNORE INTO training_data (fen, position_stats, is_tactical, difficulty)
        SELECT fen, stats, 0, 'positional' FROM elite_db.positions
    ''')
    final_conn.commit()
    final_cursor.execute("DETACH DATABASE elite_db")

    # ৬. ইমপোর্ট Tactical Puzzles
    print("🧩 Merging 3M Tactical Puzzles (Prioritizing Accuracy)...")
    final_cursor.execute(f"ATTACH DATABASE '{local_puzzle}' AS puzzle_db")
    final_cursor.execute('''
        INSERT OR REPLACE INTO training_data (fen, best_move, is_tactical, difficulty)
        SELECT fen, best_move, 1, difficulty FROM puzzle_db.puzzles
    ''')
    final_conn.commit()
    final_cursor.execute("DETACH DATABASE puzzle_db")

    # ৭. অটোমেটিক Best Move ক্যালকুলেশন (পজিশনাল ডেটার জন্য)
    print("\n🔍 Calculating Best Moves for positional samples...")
    final_cursor.execute('SELECT COUNT(*) FROM training_data WHERE best_move IS NULL')
    count_to_update = final_cursor.fetchone()[0]

    if count_to_update > 0:
        final_cursor.execute('SELECT fen, position_stats FROM training_data WHERE best_move IS NULL')
        rows = final_cursor.fetchall()

        update_batch = []
        for fen, stats_json in tqdm(rows, desc="Updating Best Moves"):
            try:
                stats = json.loads(stats_json)
                moves = stats.get('moves', {})
                if moves:
                    # সবচেয়ে বেশি জেতা/খেলা চালটি বেছে নেওয়া
                    best = max(moves.items(), key=lambda x: x[1]['white'] + x[1]['black'] + x[1]['draw'])[0]
                    update_batch.append((best, fen))

                if len(update_batch) >= 100000:
                    final_cursor.executemany('UPDATE training_data SET best_move = ? WHERE fen = ?', update_batch)
                    final_conn.commit()
                    update_batch = []
            except:
                continue

        if update_batch:
            final_cursor.executemany('UPDATE training_data SET best_move = ? WHERE fen = ?', update_batch)
            final_conn.commit()

    # ৮. ড্রাইভ ব্যাকআপ
    final_size = final_cursor.execute('SELECT COUNT(*) FROM training_data').fetchone()[0]
    final_conn.close()

    print(f"\n💾 Saving final database to Drive (Size: {os.path.getsize(local_final)/(1024**2):.2f} MB)...")
    shutil.copy(local_final, final_drive_db)

    # চেকপয়েন্ট আপডেট
    state['final_db_path'] = final_drive_db
    state['total_samples'] = final_size
    state['stage'] = 'data_ready'
    checkpoint_mgr.save(state)

    print(f"\n{'='*60}")
    print(f"✅ SUCCESS: SYNAPSE-EDGE V3 DATABASE READY")
    print(f"{'='*60}")
    print(f"📊 Total High-Quality Positions: {final_size:,}")
    print(f"💾 Database Location: {final_drive_db}")
    print("\n🚀 Next: Starting the ResNet-10 Multi-Head Training!")

# রান
merge_databases()
```


Output:



```text

🔀 Starting Synapse-Edge Final Merge...
📥 Moving files to local VM storage for 10x speed...
🔍 Validating source tables...

♟️  Importing 5M Elite Data...
🧩 Merging 3M Tactical Puzzles (Prioritizing Accuracy)...
💓 [11:05:01] Keep-Alive Signal

🔍 Calculating Best Moves for positional samples...
Updating Best Moves: 100%|██████████| 2552023/2552023 [00:35<00:00, 71606.02it/s]

💾 Saving final database to Drive (Size: 1071.69 MB)...
💓 [11:07:01] Keep-Alive Signal
💾 Checkpoint Saved: data_ready

============================================================
✅ SUCCESS: SYNAPSE-EDGE V3 DATABASE READY
============================================================
📊 Total High-Quality Positions: 5,551,558
💾 Database Location: /content/drive/MyDrive/Chessmate_Project/Synapse_Edge/data/synapse_training_final.db

🚀 Next: Starting the ResNet-10 Multi-Head Training!
```

---

### cell final
```python
# Final Cell: Securely Uploading Synapse-Edge Merged Database to Hugging Face
# Purpose: Centralizing 5.5M+ Tactical & Elite data for high-speed multi-account training.

from huggingface_hub import HfApi
import os

# --- ১. ক্রেডেনশিয়ালস এবং কনফিগারেশন ---
# আপনার Hugging Face 'Write' টোকেনটি এখানে বসান
HF_TOKEN = "HF_TOEN"

# আপনার অর্গানাইজেশন বা ইউজারনেম এবং নতুন ডেটাসেট রিপোজিটরির নাম
HF_ORG = "GambitFlow"
DATASET_NAME = "Synapse-Edge-Data"
REPO_ID = f"{HF_ORG}/{DATASET_NAME}"

# ক্লাউডের তৈরি করা মার্জড ডেটাবেসের সঠিক পাথ
db_source_path = '/content/drive/MyDrive/Chessmate_Project/Synapse_Edge/data/synapse_training_final.db'

api = HfApi(token=HF_TOKEN)

print(f"🚀 Initializing Upload Sequence for: {REPO_ID}")
print(f"📦 Source File: {db_source_path}")

if not os.path.exists(db_source_path):
    print(f"❌ ERROR: Database file not found at {db_source_path}. Please verify the path.")
else:
    file_size = os.path.getsize(db_source_path) / (1024**3) # Convert to GB
    print(f"⚖️  Database Size: {file_size:.2f} GB")

    try:
        # ২. রিপোজিটরি তৈরি (যদি আগে থেকে না থাকে)
        print("🔍 Checking repository existence...")
        api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

        # ৩. ফাইল আপলোড (Streaming for Large Files)
        print("⏳ Uploading to Hugging Face... This may take several minutes.")
        future_url = api.upload_file(
            path_or_fileobj=db_source_path,
            path_in_repo="synapse_training_final.db",
            repo_id=REPO_ID,
            repo_type="dataset"
        )

        print("\n" + "="*60)
        print("✅ SUCCESS: SYNAPSE-EDGE DATASET IS NOW CLOUD-READY!")
        print("="*60)
        print(f"🔗 Public URL: https://huggingface.co/datasets/{REPO_ID}")
        print(f"🎯 Direct Link: https://huggingface.co/datasets/{REPO_ID}/resolve/main/synapse_training_final.db")
        print("="*60)

    except Exception as e:
        print(f"\n❌ CRITICAL UPLOAD ERROR: {e}")
        print("Tip: Ensure your HF_TOKEN has 'WRITE' permissions.")
```

Output:

```text

🚀 Initializing Upload Sequence for: GambitFlow/Synapse-Edge-Data
📦 Source File: /content/drive/MyDrive/Chessmate_Project/Synapse_Edge/data/synapse_training_final.db
⚖️  Database Size: 1.05 GB
🔍 Checking repository existence...
⏳ Uploading to Hugging Face... This may take several minutes.
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
Processing Files (1 / 1)      : 100%
 1.12GB / 1.12GB, 96.0MB/s  
New Data Upload               : 100%
 1.12GB / 1.12GB, 96.0MB/s  
  ...synapse_training_final.db: 100%
 1.12GB / 1.12GB            

============================================================
✅ SUCCESS: SYNAPSE-EDGE DATASET IS NOW CLOUD-READY!
============================================================
🔗 Public URL: https://huggingface.co/datasets/GambitFlow/Synapse-Edge-Data
🎯 Direct Link: https://huggingface.co/datasets/GambitFlow/Synapse-Edge-Data/resolve/main/synapse_training_final.db
============================================================
```
---
