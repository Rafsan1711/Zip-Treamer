## Notebook Name: GambitFlow_Tactical_Forge.ipynb

---

## 1. Introduction
eta amar  puzzle er  notebook. 

---

### Cell 1: Environment Setup
```python


# Cell 1: Environment Setup & Shared Workspace Initialization
# ==============================================================================
# ⚔️ GAMBITFLOW TACTICAL FORGE - INIT
# ==============================================================================

import os
import time
import threading
from google.colab import drive

# ১. লাইব্রেরি ইন্সটল
print("⚙️ Installing Dependencies (Tactical Node)...")
!pip install python-chess zstandard

import chess
import zstandard as zstd

# ২. ড্রাইভ মাউন্ট
print("\n🔗 Connecting to Google Drive...")
drive.mount('/content/drive')

# ৩. Shared Folder কানেকশন
# ৩টি নোটবুকই এই ফোল্ডারে ডেটা সেভ করবে
PROJECT_ROOT = '/content/drive/MyDrive/GambitFlow_Project'
DATA_FACTORY_DIR = os.path.join(PROJECT_ROOT, 'Synapse_Data_Factory')

if not os.path.exists(DATA_FACTORY_DIR):
    os.makedirs(DATA_FACTORY_DIR, exist_ok=True)
    print(f"✅ Workspace Created: {DATA_FACTORY_DIR}")
else:
    print(f"✅ Connected to Workspace: {DATA_FACTORY_DIR}")

# ৪. কিপ-এলাইভ সিস্টেম
def keep_colab_awake():
    while True:
        time.sleep(60)
threading.Thread(target=keep_colab_awake, daemon=True).start()
print("✅ Keep-Alive Active. Tactical Forge Ready.")



        

```


Output:



```text
⚙️ Installing Dependencies (Tactical Node)...
Requirement already satisfied: python-chess in /usr/local/lib/python3.12/dist-packages (1.999)
Requirement already satisfied: zstandard in /usr/local/lib/python3.12/dist-packages (0.25.0)
Requirement already satisfied: chess<2,>=1 in /usr/local/lib/python3.12/dist-packages (from python-chess) (1.11.2)

🔗 Connecting to Google Drive...
Drive already mounted at /content/drive; to attempt to forcibly remount, call drive.mount("/content/drive", force_remount=True).
✅ Workspace Created: /content/drive/MyDrive/GambitFlow_Project/Synapse_Data_Factory
✅ Keep-Alive Active. Tactical Forge Ready.


```

---

### Cell 2
```python



# Cell 2: Data Acquisition (Tactical Puzzles Source)
# ==============================================================================
# 📥 DOWNLOAD RAW PUZZLE DATA (Lichess Open Database)
# ==============================================================================

import requests
import shutil
import os

# ১. কনফিগারেশন
PUZZLE_URL = "https://database.lichess.org/lichess_db_puzzle.csv.zst"
FILENAME = "lichess_db_puzzle.csv.zst"

# পাথ সেটআপ
LOCAL_DIR = "/content/data"
LOCAL_FILE_PATH = os.path.join(LOCAL_DIR, FILENAME)
DRIVE_BACKUP_PATH = os.path.join(DATA_FACTORY_DIR, FILENAME)

os.makedirs(LOCAL_DIR, exist_ok=True)

print(f"🎯 Target File: {FILENAME}")
print(f"📂 Local Path: {LOCAL_FILE_PATH}")

# ২. ডাউনলোড লজিক
if os.path.exists(LOCAL_FILE_PATH):
    print("✅ File already exists locally! Ready to forge tactics.")

elif os.path.exists(DRIVE_BACKUP_PATH):
    print("📦 Found in Shared Drive. Copying to Local SSD...")
    shutil.copy(DRIVE_BACKUP_PATH, LOCAL_FILE_PATH)
    print("✅ Copy Complete!")

else:
    print(f"⬇️ Downloading Puzzle Database (Compact yet dense)...")
    try:
        with requests.get(PUZZLE_URL, stream=True) as r:
            r.raise_for_status()
            with open(LOCAL_FILE_PATH, 'wb') as f:
                total_dl = 0
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    total_dl += len(chunk)
                    if total_dl % (50 * 1024 * 1024) == 0: # প্রতি 50MB পর আপডেট
                        print(f"   Downloaded: {total_dl / (1024*1024):.0f} MB...")

        print("✅ Download Complete!")
        # ব্যাকআপ
        # shutil.copy(LOCAL_FILE_PATH, DRIVE_BACKUP_PATH)

    except Exception as e:
        print(f"❌ Download Failed: {e}")

print(f"⚖️ File Size: {os.path.getsize(LOCAL_FILE_PATH) / (1024*1024):.2f} MB")
     
            



```


Output:



```text

🎯 Target File: lichess_db_puzzle.csv.zst
📂 Local Path: /content/data/lichess_db_puzzle.csv.zst
⬇️ Downloading Puzzle Database (Compact yet dense)...
   Downloaded: 50 MB...
   Downloaded: 100 MB...
   Downloaded: 150 MB...
   Downloaded: 200 MB...
   Downloaded: 250 MB...
✅ Download Complete!
⚖️ File Size: 270.37 MB
```

---

### Cell 3
```python

# Cell 3: Tactical Extraction & Filtering (Rating 2000+)
# ==============================================================================
# ⚔️ EXTRACT & FILTER HIGH-QUALITY PUZZLES
# ==============================================================================

import sqlite3
import csv
import io
import zstandard as zstd
import os
import shutil

# ১. কনফিগারেশন
LOCAL_DB_NAME = "tactical_puzzles.db"
DRIVE_DB_PATH = os.path.join(DATA_FACTORY_DIR, LOCAL_DB_NAME)
MIN_RATING = 2000
MIN_POPULARITY = 90  # উচ্চমানের পাজল নিশ্চিত করতে

print(f"⏳ Processing Puzzles (Filter: Rating > {MIN_RATING})...")

# ২. SQLite ডেটাবেস সেটআপ
conn = sqlite3.connect(LOCAL_DB_NAME)
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS puzzles (
        puzzle_id TEXT PRIMARY KEY,
        fen TEXT,
        moves TEXT,
        rating INTEGER,
        themes TEXT
    )
''')
cursor.execute('PRAGMA synchronous = OFF')
cursor.execute('PRAGMA journal_mode = MEMORY')

# ৩. প্রসেসিং লজিক (Streaming Decompression)
dctx = zstd.ZstdDecompressor()
puzzles_found = 0

with open(LOCAL_FILE_PATH, 'rb') as ifh:
    with dctx.stream_reader(ifh) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')
        csv_reader = csv.reader(text_stream)

        # Header স্কিপ করা
        next(csv_reader)

        batch = []
        for row in csv_reader:
            # CSV Columns: PuzzleId, FEN, Moves, Rating, RatingDeviation, Popularity, NbPlays, Themes, GameUrl, OpeningTags
            try:
                p_id, fen, moves, rating, _, pop, _, themes, _, _ = row
                rating = int(rating)
                pop = int(pop)

                # ফিল্টারিং
                if rating >= MIN_RATING and pop >= MIN_POPULARITY:
                    batch.append((p_id, fen, moves, rating, themes))
                    puzzles_found += 1

                # ব্যাচ ইনসার্ট (স্পিডের জন্য)
                if len(batch) >= 5000:
                    cursor.executemany('INSERT OR IGNORE INTO puzzles VALUES (?, ?, ?, ?, ?)', batch)
                    conn.commit()
                    batch = []
                    if puzzles_found % 10000 == 0:
                        print(f"   Stored {puzzles_found:,} hard puzzles...")
            except:
                continue

        # অবশিষ্ট ডেটা সেভ
        if batch:
            cursor.executemany('INSERT OR IGNORE INTO puzzles VALUES (?, ?, ?, ?, ?)', batch)
            conn.commit()

conn.close()
print(f"✅ Extraction Complete! Total Hard Puzzles Found: {puzzles_found:,}")

# ৪. ড্রাইভে সেভ করা
print("🚚 Moving database to Shared Drive...")
shutil.copy(LOCAL_DB_NAME, DRIVE_DB_PATH)
print(f"✅ Database Secured at: {DRIVE_DB_PATH}")

 
   

```


Output:



```text

⏳ Processing Puzzles (Filter: Rating > 2000)...
   Stored 10,000 hard puzzles...
   Stored 20,000 hard puzzles...
   Stored 30,000 hard puzzles...
   Stored 40,000 hard puzzles...
   Stored 50,000 hard puzzles...
   Stored 60,000 hard puzzles...
   Stored 70,000 hard puzzles...
   Stored 80,000 hard puzzles...
   Stored 90,000 hard puzzles...
   Stored 100,000 hard puzzles...
   Stored 110,000 hard puzzles...
   Stored 120,000 hard puzzles...
   Stored 130,000 hard puzzles...
   Stored 140,000 hard puzzles...
   Stored 150,000 hard puzzles...
   Stored 160,000 hard puzzles...
   Stored 170,000 hard puzzles...
   Stored 180,000 hard puzzles...
   Stored 190,000 hard puzzles...
   Stored 200,000 hard puzzles...
   Stored 210,000 hard puzzles...
   Stored 220,000 hard puzzles...
   Stored 230,000 hard puzzles...
   Stored 240,000 hard puzzles...
   Stored 250,000 hard puzzles...
   Stored 260,000 hard puzzles...
   Stored 270,000 hard puzzles...
   Stored 280,000 hard puzzles...
   Stored 290,000 hard puzzles...
   Stored 300,000 hard puzzles...
   Stored 310,000 hard puzzles...
   Stored 320,000 hard puzzles...
   Stored 330,000 hard puzzles...
   Stored 340,000 hard puzzles...
   Stored 350,000 hard puzzles...
   Stored 360,000 hard puzzles...
   Stored 370,000 hard puzzles...
   Stored 380,000 hard puzzles...
   Stored 390,000 hard puzzles...
   Stored 400,000 hard puzzles...
   Stored 410,000 hard puzzles...
   Stored 420,000 hard puzzles...
   Stored 430,000 hard puzzles...
   Stored 440,000 hard puzzles...
   Stored 450,000 hard puzzles...
   Stored 460,000 hard puzzles...
   Stored 470,000 hard puzzles...
   Stored 480,000 hard puzzles...
   Stored 490,000 hard puzzles...
   Stored 500,000 hard puzzles...
✅ Extraction Complete! Total Hard Puzzles Found: 500,616
🚚 Moving database to Shared Drive...
✅ Database Secured at: /content/drive/MyDrive/GambitFlow_Project/Synapse_Data_Factory/tactical_puzzles.db

  
```

