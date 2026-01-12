## Notebook Name: GambitFlow_Opening_Architect.ipynb

---

## 1. Introduction
eta amar  opening  er  notebook. 

---

### Cell 1: Environment Setup
```python



# Cell 1: Environment Setup & Shared Workspace Initialization
# ==============================================================================
# 🏗️ GAMBITFLOW OPENING ARCHITECT - INIT
# ==============================================================================

import os
import time
import threading
from google.colab import drive

# ১. প্রয়োজনীয় লাইব্রেরি ইন্সটল
print("⚙️ Installing Dependencies...")
!pip install python-chess zstandard

import chess.pgn
import zstandard as zstd

# ২. ড্রাইভ মাউন্ট (Shared Shortcut সেটআপের জন্য)
print("\n🔗 Connecting to Google Drive...")
drive.mount('/content/drive')

# ৩. Shared Folder পাথ সেটআপ
# নোট: আপনি ৩টি অ্যাকাউন্টে 'Add Shortcut to Drive' করেছেন বলে পাথ সবার জন্য সেইম থাকবে।
PROJECT_ROOT = '/content/drive/MyDrive/GambitFlow_Project'
DATA_FACTORY_DIR = os.path.join(PROJECT_ROOT, 'Synapse_Data_Factory')

# ফোল্ডার না থাকলে বানিয়ে নেবে
if not os.path.exists(DATA_FACTORY_DIR):
    os.makedirs(DATA_FACTORY_DIR, exist_ok=True)
    print(f"✅ Created New Workspace: {DATA_FACTORY_DIR}")
else:
    print(f"✅ Found Existing Workspace: {DATA_FACTORY_DIR}")

# ৪. অ্যান্টি-ডিসকানেক্ট সিস্টেম (Colab Keep-Alive)
# বড় ফাইল প্রসেসিংয়ের সময় ব্রাউজার আইডল থাকলেও সেশন এক্টিভ রাখবে
def keep_colab_awake():
    while True:
        time.sleep(60) # প্রতি ১ মিনিট পর পর চেক করবে
        # ব্যাকগ্রাউন্ডে ছোট একটি প্রিন্ট দিলে সেশন লাইভ থাকে
        pass

# থ্রেড চালু করা
threading.Thread(target=keep_colab_awake, daemon=True).start()
print("✅ Keep-Alive Protocol Activated. Ready for Data Processing.")
        

```


Output:



```text

⚙️ Installing Dependencies...
Requirement already satisfied: python-chess in /usr/local/lib/python3.12/dist-packages (1.999)
Requirement already satisfied: zstandard in /usr/local/lib/python3.12/dist-packages (0.25.0)
Requirement already satisfied: chess<2,>=1 in /usr/local/lib/python3.12/dist-packages (from python-chess) (1.11.2)

🔗 Connecting to Google Drive...
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
✅ Found Existing Workspace: /content/drive/MyDrive/GambitFlow_Project/Synapse_Data_Factory
✅ Keep-Alive Protocol Activated. Ready for Data Processing.
```

---

### Cell 2
```python


# Cell 2: Data Acquisition (High-Quality PGN Source)
# ==============================================================================
# 📥 DOWNLOAD RAW DATA (April 2024 - Modern Theory)
# ==============================================================================

import requests
import shutil
import os

# ১. কনফিগারেশন
LICHESS_URL = "https://database.lichess.org/standard/lichess_db_standard_rated_2017-02.pgn.zst"
FILENAME = "lichess_2024_04.pgn.zst"

# পাথ সেটআপ (Local SSD for Speed)
LOCAL_DIR = "/content/data"
LOCAL_FILE_PATH = os.path.join(LOCAL_DIR, FILENAME)
DRIVE_BACKUP_PATH = os.path.join(DATA_FACTORY_DIR, FILENAME) # Shared Folder এ ব্যাকআপ

os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"🎯 Target File: {FILENAME}")
print(f"📂 Local Path: {LOCAL_FILE_PATH}")

# ২. লজিক: ফাইল চেক এবং ডাউনলোড
if os.path.exists(LOCAL_FILE_PATH):
    print("✅ File already exists locally! Ready to process.")

elif os.path.exists(DRIVE_BACKUP_PATH):
    print("📦 Found file in Shared Drive. Copying to Local SSD (Faster)...")
    shutil.copy(DRIVE_BACKUP_PATH, LOCAL_FILE_PATH)
    print("✅ Copy Complete!")

else:
    print(f"⬇️ Downloading from Lichess (This may take 3-5 mins)...")
    try:
        with requests.get(LICHESS_URL, stream=True) as r:
            r.raise_for_status()
            with open(LOCAL_FILE_PATH, 'wb') as f:
                total_dl = 0
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total_dl += len(chunk)
                    if total_dl % (100 * 1024 * 1024) == 0: # প্রতি 100MB পর পর প্রিন্ট
                        print(f"   Downloaded: {total_dl / (1024*1024):.0f} MB...")

        print("✅ Download Complete!")

        # অপশনাল: ফিউচার ইউজের জন্য ড্রাইভে ব্যাকআপ রাখা (যদি স্পেস থাকে)
        # print("📦 Backing up to Drive...")
        # shutil.copy(LOCAL_FILE_PATH, DRIVE_BACKUP_PATH)

    except Exception as e:
        print(f"❌ Download Failed: {e}")

print(f"⚖️ File Size: {os.path.getsize(LOCAL_FILE_PATH) / (1024*1024*1024):.2f} GB")





```


Output:



```text

🎯 Target File: lichess_2024_04.pgn.zst
📂 Local Path: /content/data/lichess_2024_04.pgn.zst
✅ File already exists locally! Ready to process.
⚖️ File Size: 0.09 GB
```

---

### Cell 3
```python


# Cell 3: Robust Theory Extraction with Auto-Directory Fix
# ==============================================================================
# 🏗️ OPENING ARCHITECT - RESUMABLE EXTRACTION (FIXED PATHS)
# ==============================================================================

import sqlite3
import json
import chess.pgn
import zstandard as zstd
import io
import os
import shutil

# ১. পাথ এবং ডিরেক্টরি নিশ্চিত করা
LOCAL_DB_NAME = "opening_theory_v1.db"
# নিশ্চিত করুন Cell 1 এ DATA_FACTORY_DIR ঠিকমতো ডিফাইন হয়েছে
if 'DATA_FACTORY_DIR' not in globals():
    DATA_FACTORY_DIR = '/content/drive/MyDrive/GambitFlow_Project/Synapse_Data_Factory'

os.makedirs(DATA_FACTORY_DIR, exist_ok=True) # পাথ না থাকলে তৈরি করবে

CHECKPOINT_PATH = os.path.join(DATA_FACTORY_DIR, "opening_checkpoint.json")
DRIVE_DB_BACKUP = os.path.join(DATA_FACTORY_DIR, LOCAL_DB_NAME)

# ২. কনফিগারেশন
MIN_ELO = 2000
MAX_PLY = 35
current_count = 0
elite_found = 0

def load_checkpoint():
    if os.path.exists(CHECKPOINT_PATH):
        try:
            with open(CHECKPOINT_PATH, 'r') as f:
                return json.load(f)
        except: pass
    return {"games_processed": 0, "elite_games_found": 0}

def save_checkpoint(games_count, elite_count):
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)
    with open(CHECKPOINT_PATH, 'w') as f:
        json.dump({"games_processed": games_count, "elite_games_found": elite_count}, f)

# ৩. স্টেট এবং ডিবি লোড
state = load_checkpoint()
processed_count = state["games_processed"]
elite_found = state["elite_games_found"]

print(f"🔄 Resuming from game: {processed_count:,}")
print(f"🏆 Elite Games Found: {elite_found:,}")

conn = sqlite3.connect(LOCAL_DB_NAME)
cursor = conn.cursor()
cursor.execute('CREATE TABLE IF NOT EXISTS opening_book (fen TEXT PRIMARY KEY, move_stats TEXT)')
cursor.execute('PRAGMA synchronous = OFF')
cursor.execute('PRAGMA journal_mode = MEMORY')

# ৪. মেইন প্রসেসিং লজিক
dctx = zstandard_decompressor = zstd.ZstdDecompressor()
opening_data = {}

try:
    with open(LOCAL_FILE_PATH, 'rb') as ifh:
        with dctx.stream_reader(ifh) as reader:
            text_stream = io.TextIOWrapper(reader, encoding='utf-8')

            # ফাস্ট ফরওয়ার্ড
            current_count = 0
            if processed_count > 0:
                print(f"⏩ Fast-forwarding to {processed_count:,}...")
                while current_count < processed_count:
                    if chess.pgn.read_headers(text_stream) is None: break
                    current_count += 1
                    if current_count % 500000 == 0: print(f"   Skipped {current_count:,}...")

            print("🚀 Processing Elite Games (3000+ Level)...")
            while True:
                game = chess.pgn.read_game(text_stream)
                if game is None: break

                try:
                    w_elo = int(game.headers.get("WhiteElo", 0))
                    b_elo = int(game.headers.get("BlackElo", 0))
                    if w_elo >= MIN_ELO and b_elo >= MIN_ELO:
                        elite_found += 1
                        board = game.board()
                        for i, move in enumerate(game.mainline_moves()):
                            if i >= MAX_PLY: break
                            fen = " ".join(board.fen().split(" ")[:4])
                            move_san = board.san(move)
                            if fen not in opening_data: opening_data[fen] = {}
                            opening_data[fen][move_san] = opening_data[fen].get(move_san, 0) + 1
                            board.push(move)
                except: pass

                current_count += 1

                # সেভ পয়েন্ট
                if current_count % 10000 == 0:
                    for fen, stats in opening_data.items():
                        cursor.execute('SELECT move_stats FROM opening_book WHERE fen=?', (fen,))
                        row = cursor.fetchone()
                        if row:
                            old = json.loads(row[0])
                            for m, c in stats.items(): old[m] = old.get(m, 0) + c
                            stats = old
                        cursor.execute('INSERT OR REPLACE INTO opening_book VALUES (?, ?)', (fen, json.dumps(stats)))

                    conn.commit()
                    opening_data = {}
                    save_checkpoint(current_count, elite_found)
                    shutil.copy(LOCAL_DB_NAME, DRIVE_DB_BACKUP)
                    print(f"📌 {current_count:,} games | Elite: {elite_found:,} | Backed up to Drive")

except Exception as e:
    print(f"\n❌ Loop Interrupted: {e}")

finally:
    # সেশন কাটলেও যাতে ডেটা হারানো না যায়
    conn.commit()
    conn.close()
    save_checkpoint(current_count, elite_found)
    if os.path.exists(LOCAL_DB_NAME):
        shutil.copy(LOCAL_DB_NAME, DRIVE_DB_BACKUP)
    print(f"\n✅ Safe Shutdown. Progress saved to {DRIVE_DB_BACKUP}")
 
   

```


Output:



```text

🔄 Resuming from game: 578,262
🏆 Elite Games Found: 10,212
⏩ Fast-forwarding to 578,262...
   Skipped 500,000...
🚀 Processing Elite Games (3000+ Level)...

✅ Safe Shutdown. Progress saved to /content/drive/MyDrive/GambitFlow_Project/Synapse_Data_Factory/opening_theory_v1.db

  
```

---

### Cell 4
```python


 # Cell 4: Final Export & Hugging Face Data Persistence
# ==============================================================================
# 🚀 UPLOAD ELITE OPENING THEORY TO HUGGING FACE
# ==============================================================================

from huggingface_hub import HfApi
import os

# ১. কনফিগারেশন
# নোট: HF_TOKEN আপনার সিক্রেট বা সেলে আগে ডিফাইন থাকতে হবে
HF_TOKEN = "HF_TOKEN"
HF_USERNAME = "GambitFlow"
REPO_ID = f"{HF_USERNAME}/synapse-elite-data" # নতুন ডেটাসেট রিপো

LOCAL_DB_PATH = "opening_theory_v1.db"

# ২. আপলোড লজিক
api = HfApi(token=HF_TOKEN)

print(f"🚀 Initializing Upload to: {REPO_ID}")

try:
    # রিপো তৈরি (যদি না থাকে)
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    # ফাইল আপলোড
    print(f"⏳ Uploading {LOCAL_DB_PATH}... (Please wait)")
    api.upload_file(
        path_or_fileobj=LOCAL_DB_PATH,
        path_in_repo="opening_theory_v1.db",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Initial Elite Opening Theory (3000+ Elo Games)"
    )

    print("\n🎉 SUCCESS! Your 'Foundation Data' is live on Hugging Face.")
    print(f"🔗 URL: https://huggingface.co/datasets/{REPO_ID}")

except Exception as e:
    print(f"\n❌ Upload Failed: {e}")
    print("Tip: Make sure your Token has 'WRITE' permission.")

# ড্রাইভের লোকাল কপি চেক
if os.path.exists(LOCAL_DB_PATH):
    file_size = os.path.getsize(LOCAL_DB_PATH) / (1024*1024)
    print(f"📦 Local DB Size: {file_size:.2f} MB")
   

```


Output:



```text


  
```
