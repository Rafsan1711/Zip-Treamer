## Notebook Name: Chessmate Data Prep.ipynb

---

## 1. Introduction
eta amar  GambitFlow/Nexus-Nano ei model er dataset bananor notebook. 

---

## 2. Code Implementation

### cell 1
```python

from google.colab import drive
import os

# ১. ড্রাইভ মাউন্ট করা (কানেক্ট করা)
# এটা রান করলে একটি পপ-আপ আসবে, সেখানে 'Allow' বা অনুমতি দিতে হবে।
drive.mount('/content/drive')

# ২. আমাদের প্রোজেক্টের জন্য ড্রাইভে একটা ফোল্ডার বানানো
# এই ফোল্ডারেই সব ফাইল সেভ হবে।
project_folder = '/content/drive/MyDrive/Chessmate_Project'
os.makedirs(project_folder, exist_ok=True)

print(f"Project Folder Ready at: {project_folder}")
```
Output:

```text
Mounted at /content/drive
Project Folder Ready at: /content/drive/MyDrive/Chessmate_Project
```


---


### cell 2
```python
!pip install python-chess zstandard
```

Output:

```text
Collecting python-chess
  Downloading python_chess-1.999-py3-none-any.whl.metadata (776 bytes)
Requirement already satisfied: zstandard in /usr/local/lib/python3.12/dist-packages (0.25.0)
Collecting chess<2,>=1 (from python-chess)
  Downloading chess-1.11.2.tar.gz (6.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.1/6.1 MB 37.6 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Downloading python_chess-1.999-py3-none-any.whl (1.4 kB)
Building wheels for collected packages: chess
  Building wheel for chess (setup.py) ... done
  Created wheel for chess: filename=chess-1.11.2-py3-none-any.whl size=147775 sha256=98a8d32398a0f039b6cd83c67352a1da28d7c3e4465151a2725f33fa06aed946
  Stored in directory: /root/.cache/pip/wheels/83/1f/4e/8f4300f7dd554eb8de70ddfed96e94d3d030ace10c5b53d447
Successfully built chess
Installing collected packages: chess, python-chess
Successfully installed chess-1.11.2 python-chess-1.999
```
---

### cell 3
```python

import chess.pgn      # দাবার গেম রিড করার জন্য
import zstandard as zstd  # .zst কম্প্রেসড ফাইল খোলার জন্য
import io             # ডেটা পড়ার স্ট্রিম বা প্রবাহ
import json           # ডেটা সেভ করার ফরম্যাট
import requests       # ইন্টারনেট থেকে ডাউনলোড করার জন্য
```
---

### cell 4
```python

# ফাইলের লিংক এবং সেভ করার লোকেশন
url = "https://database.lichess.org/standard/lichess_db_standard_rated_2016-02.pgn.zst"
output_file = os.path.join(project_folder, "lichess_data.pgn.zst")

print(f"Checking if file exists at: {output_file}")

# যদি ফাইল আগে থেকেই থাকে, তাহলে পুনরায় ডাউনলোড করব না (সময় বাঁচবে)
if os.path.exists(output_file):
    print("File already exists! Skipping download.")
else:
    print("Downloading database... (This is ~900MB, might take 2-5 minutes)")
    response = requests.get(url, stream=True)

    # ফাইল রাইট করা হচ্ছে
    with open(output_file, "wb") as f:
        for chunk in response.iter_content(chunk_size=1024*1024): # ১ মেগাবাইট করে নামবে
            if chunk:
                f.write(chunk)
    print("Download complete!")
```

Output:


```text
Checking if file exists at: /content/drive/MyDrive/Chessmate_Project/lichess_data.pgn.zst
Downloading database... (This is ~900MB, might take 2-5 minutes)
Download complete!
```

---

### cell 5
```python

# আমরা টেস্ট করার জন্য আপাতত ১ লক্ষ গেম প্রসেস করব
MAX_GAMES = 100000
stats = {} # এখানেই সব তথ্য জমা হবে (Python Dictionary)

print("Processing games from Drive... Please wait.")

# Zstandard ডিকম্প্রেসার সেটআপ
dctx = zstd.ZstdDecompressor()

# ফাইলটি ওপেন করা হচ্ছে
with open(output_file, 'rb') as ifh:
    with dctx.stream_reader(ifh) as reader:
        text_stream = io.TextIOWrapper(reader, encoding='utf-8')

        game_count = 0

        while game_count < MAX_GAMES:
            # একটি গেম পড়া
            try:
                game = chess.pgn.read_game(text_stream)
            except Exception as e:
                print(f"Error reading game: {e}")
                continue

            if game is None:
                break

            game_count += 1
            if game_count % 10000 == 0:
                print(f"Processed {game_count} games...")

            # ১. রেজাল্ট চেক করা
            result = game.headers.get("Result")
            if result not in ['1-0', '0-1', '1/2-1/2']:
                continue

            board = game.board()

            # ২. চালগুলো পড়া (প্রথম ১৫ চাল পর্যন্ত)
            for i, move in enumerate(game.mainline_moves()):
                if i > 15: # ১৫ চালের পর আর দরকার নেই
                    break

                fen = board.fen()
                move_san = board.san(move)

                # FEN ক্লিন করা (En passant এবং move number বাদ দেওয়া)
                # মূল FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1
                # ক্লিন FEN: rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq
                clean_fen = " ".join(fen.split(" ")[:4])

                # ৩. স্ট্যাটস আপডেট করা
                if clean_fen not in stats:
                    stats[clean_fen] = {"total": 0, "moves": {}}

                stats[clean_fen]["total"] += 1

                if move_san not in stats[clean_fen]["moves"]:
                    stats[clean_fen]["moves"][move_san] = {"white": 0, "black": 0, "draw": 0}

                if result == '1-0': # সাদা জয়ী
                    stats[clean_fen]["moves"][move_san]["white"] += 1
                elif result == '0-1': # কালো জয়ী
                    stats[clean_fen]["moves"][move_san]["black"] += 1
                else: # ড্র
                    stats[clean_fen]["moves"][move_san]["draw"] += 1

                # বোর্ডে চাল দেওয়া
                board.push(move)

print(f"Finished! Total unique positions found: {len(stats)}")
```

Output:

```text
Processing games from Drive... Please wait.
Processed 10000 games...
Processed 20000 games...
Processed 30000 games...
Processed 40000 games...
Processed 50000 games...
Processed 60000 games...
Processed 70000 games...
Processed 80000 games...
Processed 90000 games...
Processed 100000 games...
Finished! Total unique positions found: 638018
```

---

### Cell 6
```python

json_output_path = os.path.join(project_folder, 'opening_stats.json')

with open(json_output_path, 'w') as f:
    json.dump(stats, f)

print(f"SUCCESS! JSON file saved at: {json_output_path}")
print("You can now go to your Google Drive folder 'Chessmate_Project' and see the file.")
```


Output:



```text
SUCCESS! JSON file saved at: /content/drive/MyDrive/Chessmate_Project/opening_stats.json
You can now go to your Google Drive folder 'Chessmate_Project' and see the file.
```

---

### Cell 7
```python
!pip install huggingface_hub
```



Output:


```text
Requirement already satisfied: huggingface_hub in /usr/local/lib/python3.12/dist-packages (0.36.0)
Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (3.20.0)
Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (2025.3.0)
Requirement already satisfied: packaging>=20.9 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (25.0)
Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (6.0.3)
Requirement already satisfied: requests in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (2.32.4)
Requirement already satisfied: tqdm>=4.42.1 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (4.67.1)
Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (4.15.0)
Requirement already satisfied: hf-xet<2.0.0,>=1.1.3 in /usr/local/lib/python3.12/dist-packages (from huggingface_hub) (1.2.0)
Requirement already satisfied: charset_normalizer<4,>=2 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface_hub) (3.4.4)
Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface_hub) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface_hub) (2.5.0)
Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.12/dist-packages (from requests->huggingface_hub) (2025.11.12)
```
---

### Cell 8
```python
import os
import shutil
from google.colab import drive

# ১. ড্রাইভ আবার কানেক্ট করি (নিশ্চিত হওয়ার জন্য)
drive.mount('/content/drive')

# আমাদের টার্গেট ফোল্ডার এবং ফাইল পাথ
target_folder = '/content/drive/MyDrive/Chessmate_Project'
target_path = os.path.join(target_folder, 'opening_stats.json')
local_path = '/content/opening_stats.json'  # যদি ভুলে বাইরে সেভ হয়ে থাকে

# ফোল্ডার না থাকলে বানিয়ে নিবে
os.makedirs(target_folder, exist_ok=True)

print("Checking file location...")

if os.path.exists(target_path):
    print(f"✅ Great! File found correctly at: {target_path}")
    print(f"File size: {os.path.getsize(target_path) / (1024*1024):.2f} MB")

elif os.path.exists(local_path):
    print(f"⚠️ File found in local storage, moving it to Drive...")
    shutil.move(local_path, target_path)
    print(f"✅ Moved successfully to: {target_path}")

else:
    print("❌ DANGER: File not found anywhere!")
    print("আপনাকে সম্ভবত 'ধাপ ৫' (Game Processing) আবার রান করতে হবে কারণ ফাইলটি তৈরি হয়নি বা মুছে গেছে।")
```
Output:



```text
Mounted at /content/drive
Checking file location...
✅ Great! File found correctly at: /content/drive/MyDrive/Chessmate_Project/opening_stats.json
File size: 86.95 MB
```

---


### Cell 9
```python
from huggingface_hub import HfApi

# --- আপনার তথ্য ---
# ১. আপনার Write টোকেনটি ডাবল কোট এর ভেতরে দিন
HF_TOKEN = "MY-HF-TOKEN"

# ২. আপনার ইউজারনেম (আপনার দেওয়া তথ্য অনুযায়ী বসালাম)
HF_USERNAME = "Rafs-an09002"
# ------------------

REPO_ID = f"{HF_USERNAME}/chessmate-opening-stats"
file_path = '/content/drive/MyDrive/Chessmate_Project/opening_stats.json'

api = HfApi(token=HF_TOKEN)

print(f"Target Repository: {REPO_ID}")
print("Uploading file to Hugging Face... (Might take 1-2 minutes)")

try:
    # ১. রিপোজিটরি আছে কিনা চেক করছি, না থাকলে বানিয়ে নিবে
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    # ২. ফাইল আপলোড করছি
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo="opening_stats.json",
        repo_id=REPO_ID,
        repo_type="dataset"
    )
    print("\n🎉 Congratulations! Upload Complete.")
    print(f"Check your file here: https://huggingface.co/datasets/{REPO_ID}")

except Exception as e:
    print(f"\n❌ Error: {e}")
```


Output:



```text
Target Repository: Rafs-an09002/chessmate-opening-stats
Uploading file to Hugging Face... (Might take 1-2 minutes)
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
Processing Files (1 / 1)      : 100%
 91.2MB / 91.2MB, 35.1MB/s  
New Data Upload               : 100%
 91.2MB / 91.2MB, 35.1MB/s  
  ...roject/opening_stats.json: 100%
 91.2MB / 91.2MB            

🎉 Congratulations! Upload Complete.
Check your file here: https://huggingface.co/datasets/Rafs-an09002/chessmate-opening-stats
```

---


### Cell 10
```python

import sqlite3
import json
import os
import shutil
from google.colab import drive

# ১. ড্রাইভ মাউন্ট করা (নিশ্চিত হওয়ার জন্য)
drive.mount('/content/drive')

# পাথ সেট করা
drive_folder = '/content/drive/MyDrive/Chessmate_Project'
json_path = os.path.join(drive_folder, 'opening_stats.json')

# আমরা লোকাল মেমোরিতে ডিবি বানাবো (ফাস্ট এবং এরর ফ্রি)
local_db_name = 'chess_stats.db'

print("⏳ Converting JSON to SQLite Database (Locally)...")

if not os.path.exists(json_path):
    print(f"❌ Error: JSON file not found at {json_path}")
else:
    # ১. ডেটাবেস তৈরি (লোকাল স্টোরেজে)
    conn = sqlite3.connect(local_db_name)
    cursor = conn.cursor()

    # ২. টেবিল বানানো
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS positions (
            fen TEXT PRIMARY KEY,
            stats TEXT
        )
    ''')

    # স্পিড বাড়ানোর জন্য কিছু সেটিংস
    cursor.execute('PRAGMA synchronous = OFF')
    cursor.execute('PRAGMA journal_mode = MEMORY')

    # ৩. JSON রিড করা এবং ডেটাবেসে ঢোকানো
    print("Reading JSON file...")
    with open(json_path, 'r') as f:
        data = json.load(f)

        print(f"Total positions to insert: {len(data)}")

        count = 0
        batch = []

        for fen, stats in data.items():
            batch.append((fen, json.dumps(stats)))
            count += 1

            if count % 10000 == 0:
                cursor.executemany('INSERT OR IGNORE INTO positions VALUES (?, ?)', batch)
                conn.commit()
                batch = []
                print(f"Inserted {count} positions...")

        # বাকিগুলো ঢোকানো
        if batch:
            cursor.executemany('INSERT OR IGNORE INTO positions VALUES (?, ?)', batch)
            conn.commit()

    conn.close()
    print("✅ Database created locally!")

    # ৪. ফাইলটি ড্রাইভে কপি করা
    final_db_path = os.path.join(drive_folder, local_db_name)
    print(f"🚚 Moving database to Google Drive: {final_db_path} ...")

    shutil.copy(local_db_name, final_db_path)

    print(f"✅ SUCCESS! Database saved in Drive.")
    print(f"Size: {os.path.getsize(final_db_path) / (1024*1024):.2f} MB")
```


Output:



```text
Mounted at /content/drive
⏳ Converting JSON to SQLite Database (Locally)...
Reading JSON file...
Total positions to insert: 638018
Inserted 10000 positions...
Inserted 20000 positions...
Inserted 30000 positions...
Inserted 40000 positions...
Inserted 50000 positions...
Inserted 60000 positions...
Inserted 70000 positions...
Inserted 80000 positions...
Inserted 90000 positions...
Inserted 100000 positions...
Inserted 110000 positions...
Inserted 120000 positions...
Inserted 130000 positions...
Inserted 140000 positions...
Inserted 150000 positions...
Inserted 160000 positions...
Inserted 170000 positions...
Inserted 180000 positions...
Inserted 190000 positions...
Inserted 200000 positions...
Inserted 210000 positions...
Inserted 220000 positions...
Inserted 230000 positions...
Inserted 240000 positions...
Inserted 250000 positions...
Inserted 260000 positions...
Inserted 270000 positions...
Inserted 280000 positions...
Inserted 290000 positions...
Inserted 300000 positions...
Inserted 310000 positions...
Inserted 320000 positions...
Inserted 330000 positions...
Inserted 340000 positions...
Inserted 350000 positions...
Inserted 360000 positions...
Inserted 370000 positions...
Inserted 380000 positions...
Inserted 390000 positions...
Inserted 400000 positions...
Inserted 410000 positions...
Inserted 420000 positions...
Inserted 430000 positions...
Inserted 440000 positions...
Inserted 450000 positions...
Inserted 460000 positions...
Inserted 470000 positions...
Inserted 480000 positions...
Inserted 490000 positions...
Inserted 500000 positions...
Inserted 510000 positions...
Inserted 520000 positions...
Inserted 530000 positions...
Inserted 540000 positions...
Inserted 550000 positions...
Inserted 560000 positions...
Inserted 570000 positions...
Inserted 580000 positions...
Inserted 590000 positions...
Inserted 600000 positions...
Inserted 610000 positions...
Inserted 620000 positions...
Inserted 630000 positions...
✅ Database created locally!
🚚 Moving database to Google Drive: /content/drive/MyDrive/Chessmate_Project/chess_stats.db ...
✅ SUCCESS! Database saved in Drive.
Size: 141.77 MB
```


---

### Cell 11
```python
from huggingface_hub import HfApi

# আপনার টোকেন এবং ইউজারনেম
HF_TOKEN = "MY-HF-TOKEN"
HF_USERNAME = "Rafs-an09002"

REPO_ID = f"{HF_USERNAME}/chessmate-opening-stats"
file_path = '/content/drive/MyDrive/Chessmate_Project/chess_stats.db'

api = HfApi(token=HF_TOKEN)

print("Uploading SQLite database to Hugging Face...")
print("This might take 1-2 minutes (141 MB)...")

try:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo="chess_stats.db",  # ফাইলের নাম
        repo_id=REPO_ID,
        repo_type="dataset"
    )
    print("✅ Database Uploaded Successfully!")
    print(f"URL: https://huggingface.co/datasets/{REPO_ID}/resolve/main/chess_stats.db")
except Exception as e:
    print(f"❌ Error: {e}")
```


Output:



```text
Uploading SQLite database to Hugging Face...
This might take 1-2 minutes (141 MB)...
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
Processing Files (1 / 1)      : 100%
  149MB /  149MB, 74.4MB/s  
New Data Upload               : 100%
  149MB /  149MB, 74.4MB/s  
  ...te_Project/chess_stats.db: 100%
  149MB /  149MB            
✅ Database Uploaded Successfully!
URL: https://huggingface.co/datasets/Rafs-an09002/chessmate-opening-stats/resolve/main/chess_stats.db
```

---

### Cell 12
```python

# ১. পরিবেশ সেটআপ
!pip install torch chess python-chess numpy

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import json
import random
import time
from google.colab import drive

# ২. ড্রাইভ মাউন্ট করা এবং ফাইল পাথ সেট করা
drive.mount('/content/drive')
project_folder = '/content/drive/MyDrive/Chessmate_Project'
json_path = os.path.join(project_folder, 'opening_stats.json')

# ৩. GPU চেক করা
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

```


Output:



```text
Requirement already satisfied: torch in /usr/local/lib/python3.12/dist-packages (2.9.0+cu126)
Collecting chess
  Downloading chess-1.11.2.tar.gz (6.1 MB)
     ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 6.1/6.1 MB 78.4 MB/s eta 0:00:00
  Preparing metadata (setup.py) ... done
Collecting python-chess
  Downloading python_chess-1.999-py3-none-any.whl.metadata (776 bytes)
Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (2.0.2)
Requirement already satisfied: filelock in /usr/local/lib/python3.12/dist-packages (from torch) (3.20.0)
Requirement already satisfied: typing-extensions>=4.10.0 in /usr/local/lib/python3.12/dist-packages (from torch) (4.15.0)
Requirement already satisfied: setuptools in /usr/local/lib/python3.12/dist-packages (from torch) (75.2.0)
Requirement already satisfied: sympy>=1.13.3 in /usr/local/lib/python3.12/dist-packages (from torch) (1.14.0)
Requirement already satisfied: networkx>=2.5.1 in /usr/local/lib/python3.12/dist-packages (from torch) (3.6)
Requirement already satisfied: jinja2 in /usr/local/lib/python3.12/dist-packages (from torch) (3.1.6)
Requirement already satisfied: fsspec>=0.8.5 in /usr/local/lib/python3.12/dist-packages (from torch) (2025.3.0)
Requirement already satisfied: nvidia-cuda-nvrtc-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.77)
Requirement already satisfied: nvidia-cuda-runtime-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.77)
Requirement already satisfied: nvidia-cuda-cupti-cu12==12.6.80 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.80)
Requirement already satisfied: nvidia-cudnn-cu12==9.10.2.21 in /usr/local/lib/python3.12/dist-packages (from torch) (9.10.2.21)
Requirement already satisfied: nvidia-cublas-cu12==12.6.4.1 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.4.1)
Requirement already satisfied: nvidia-cufft-cu12==11.3.0.4 in /usr/local/lib/python3.12/dist-packages (from torch) (11.3.0.4)
Requirement already satisfied: nvidia-curand-cu12==10.3.7.77 in /usr/local/lib/python3.12/dist-packages (from torch) (10.3.7.77)
Requirement already satisfied: nvidia-cusolver-cu12==11.7.1.2 in /usr/local/lib/python3.12/dist-packages (from torch) (11.7.1.2)
Requirement already satisfied: nvidia-cusparse-cu12==12.5.4.2 in /usr/local/lib/python3.12/dist-packages (from torch) (12.5.4.2)
Requirement already satisfied: nvidia-cusparselt-cu12==0.7.1 in /usr/local/lib/python3.12/dist-packages (from torch) (0.7.1)
Requirement already satisfied: nvidia-nccl-cu12==2.27.5 in /usr/local/lib/python3.12/dist-packages (from torch) (2.27.5)
Requirement already satisfied: nvidia-nvshmem-cu12==3.3.20 in /usr/local/lib/python3.12/dist-packages (from torch) (3.3.20)
Requirement already satisfied: nvidia-nvtx-cu12==12.6.77 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.77)
Requirement already satisfied: nvidia-nvjitlink-cu12==12.6.85 in /usr/local/lib/python3.12/dist-packages (from torch) (12.6.85)
Requirement already satisfied: nvidia-cufile-cu12==1.11.1.6 in /usr/local/lib/python3.12/dist-packages (from torch) (1.11.1.6)
Requirement already satisfied: triton==3.5.0 in /usr/local/lib/python3.12/dist-packages (from torch) (3.5.0)
Requirement already satisfied: mpmath<1.4,>=1.1.0 in /usr/local/lib/python3.12/dist-packages (from sympy>=1.13.3->torch) (1.3.0)
Requirement already satisfied: MarkupSafe>=2.0 in /usr/local/lib/python3.12/dist-packages (from jinja2->torch) (3.0.3)
Downloading python_chess-1.999-py3-none-any.whl (1.4 kB)
Building wheels for collected packages: chess
  Building wheel for chess (setup.py) ... done
  Created wheel for chess: filename=chess-1.11.2-py3-none-any.whl size=147775 sha256=907e057acf276ba4ef00f4997a93712268c0a96433013d182d819abf12b6f33d
  Stored in directory: /root/.cache/pip/wheels/83/1f/4e/8f4300f7dd554eb8de70ddfed96e94d3d030ace10c5b53d447
Successfully built chess
Installing collected packages: chess, python-chess
Successfully installed chess-1.11.2 python-chess-1.999
Mounted at /content/drive
Device: cuda
```

---

### Cell 13
```python


# ONNX Export এর জন্য প্রয়োজনীয় লাইব্রেরি ইন্সটল
print("Installing ONNX dependencies...")
!pip install onnxscript
print("ONNX dependencies installed.")
```


Output:



```text
Installing ONNX dependencies...
Collecting onnxscript
  Downloading onnxscript-0.5.6-py3-none-any.whl.metadata (13 kB)
Requirement already satisfied: ml_dtypes in /usr/local/lib/python3.12/dist-packages (from onnxscript) (0.5.4)
Requirement already satisfied: numpy in /usr/local/lib/python3.12/dist-packages (from onnxscript) (2.0.2)
Collecting onnx_ir<2,>=0.1.12 (from onnxscript)
  Downloading onnx_ir-0.1.12-py3-none-any.whl.metadata (3.2 kB)
Collecting onnx>=1.16 (from onnxscript)
  Downloading onnx-1.20.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
Requirement already satisfied: packaging in /usr/local/lib/python3.12/dist-packages (from onnxscript) (25.0)
Requirement already satisfied: typing_extensions>=4.10 in /usr/local/lib/python3.12/dist-packages (from onnxscript) (4.15.0)
Requirement already satisfied: protobuf>=4.25.1 in /usr/local/lib/python3.12/dist-packages (from onnx>=1.16->onnxscript) (5.29.5)
Downloading onnxscript-0.5.6-py3-none-any.whl (683 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 683.0/683.0 kB 18.4 MB/s eta 0:00:00
Downloading onnx-1.20.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (18.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 18.1/18.1 MB 122.7 MB/s eta 0:00:00
Downloading onnx_ir-0.1.12-py3-none-any.whl (129 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 129.3/129.3 kB 13.5 MB/s eta 0:00:00
Installing collected packages: onnx, onnx_ir, onnxscript
Successfully installed onnx-1.20.0 onnx_ir-0.1.12 onnxscript-0.5.6
ONNX dependencies installed.
```

---

### Cell 14
```python

# FEN থেকে 12x8x8 টেনসর তৈরির ফাংশন
def fen_to_tensor(fen):
    # 'rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 1' -> শুধু পজিশন
    position = fen.split(' ')[0]

    # 12 চ্যানেল: 6টি সাদা (P, N, B, R, Q, K) এবং 6টি কালো (p, n, b, r, q, k)
    tensor = np.zeros((12, 8, 8), dtype=np.float32)

    # পিস ম্যাপ
    piece_to_channel = {
        'P': 0, 'N': 1, 'B': 2, 'R': 3, 'Q': 4, 'K': 5,
        'p': 6, 'n': 7, 'b': 8, 'r': 9, 'q': 10, 'k': 11
    }

    rank = 0
    file = 0

    for char in position:
        if char == '/':
            # নতুন র্যাঙ্ক
            rank += 1
            file = 0
        elif char.isdigit():
            # ফাঁকা ঘর
            file += int(char)
        elif char in piece_to_channel:
            # গুটি
            channel = piece_to_channel[char]
            tensor[channel, rank, file] = 1.0
            file += 1

    return torch.from_numpy(tensor).unsqueeze(0) # (1, 12, 8, 8)

```


Output:



```text

```

---

### Cell 15
```python


# ডেটাসেট ক্লাস
class ChessDataset(Dataset):
    def __init__(self, json_path, max_positions=50000):
        self.data = []

        with open(json_path, 'r') as f:
            stats = json.load(f)

        print(f"Total unique FENs in stats file: {len(stats)}")

        # প্রতিটি FEN এবং তার প্রতিটি পরবর্তী বৈধ চালকে ট্রেইনিং ডেটা হিসেবে নেওয়া
        for fen, stat_data in stats.items():

            # এই পজিশনে সাদার চাল নাকি কালোর
            turn = fen.split(' ')[1]

            # প্রতিটি চালের জন্য ডেটা তৈরি করা
            for move_san, results in stat_data['moves'].items():

                total = results['white'] + results['black'] + results['draw']
                if total < 10: continue # ১০টার কম গেম থাকলে বাদ দিলাম (কোয়ালিটি)

                # Evaluation Score তৈরি: Win% - Loss%
                # Score = (White Win Count - Black Win Count) / Total Count
                score = (results['white'] - results['black']) / total

                # স্কোরকে -1 থেকে +1 এর মধ্যে রেখে দিলাম

                self.data.append({
                    'fen': fen,
                    'score': score
                })

        # ডেটা অনেক বড় হতে পারে, তাই স্যাম্পল করা
        if len(self.data) > max_positions:
            self.data = random.sample(self.data, max_positions)

        print(f"Total training positions selected: {len(self.data)}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        # FEN কে টেনসরে রূপান্তর
        tensor = fen_to_tensor(item['fen']).squeeze(0) # (12, 8, 8)
        score = torch.tensor([item['score']], dtype=torch.float32)

        return tensor.to(device), score.to(device)

# ডেটাসেট তৈরি করা
train_data = ChessDataset(json_path, max_positions=100000) # ১ লাখ স্যাম্পল
train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
print("Dataset and DataLoader ready.")
```


Output:



```text
Total unique FENs in stats file: 638018
Total training positions selected: 9522
Dataset and DataLoader ready.
```

---

### Cell 16
```python

# মডেল আর্কিটেকচার
class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()

        # ইনপুট: (12, 8, 8)
        self.conv_layer = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        # আউটপুট কনভোলিউশনের পর: (128, 8, 8)

        # 128 * 8 * 8 = 8192 ফিচারকে ফ্ল্যাট করা
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1) # ফাইনাল আউটপুট: 1 (Evaluation Score)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)  # ফ্ল্যাট করা
        x = self.fc_layer(x)
        # tanh ব্যবহার করে আউটপুটকে [-1, 1] রেঞ্জে রাখা (Evaluation Score)
        return torch.tanh(x)

model = ChessNet().to(device)
print("Model created.")

```


Output:



```text
Model created.
```

---

### Cell 17
```python

# ট্রেইনিং কনফিগারেশন
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# MSELoss ব্যবহার করব কারণ এটা Regression (মান অনুমান) সমস্যা
criterion = nn.MSELoss()
EPOCHS = 10

print(f"Starting Training for {EPOCHS} epochs...")

for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    start_time = time.time()

    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    avg_loss = running_loss / len(train_loader)
    end_time = time.time()

    print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {avg_loss:.6f} | Time: {end_time - start_time:.2f}s")

print("Training finished.")

# মডেল সেভ করা
model_path = os.path.join(project_folder, 'best_chess_model.pth')
torch.save(model.state_dict(), model_path)
print(f"Model saved to Drive: {model_path}")

```


Output:



```text
Starting Training for 10 epochs...
Epoch 1/10 | Loss: 0.100259 | Time: 1.35s
Epoch 2/10 | Loss: 0.074322 | Time: 1.32s
Epoch 3/10 | Loss: 0.072916 | Time: 1.32s
Epoch 4/10 | Loss: 0.072781 | Time: 1.62s
Epoch 5/10 | Loss: 0.072873 | Time: 2.58s
Epoch 6/10 | Loss: 0.071655 | Time: 1.36s
Epoch 7/10 | Loss: 0.070856 | Time: 1.31s
Epoch 8/10 | Loss: 0.069634 | Time: 1.31s
Epoch 9/10 | Loss: 0.068775 | Time: 1.31s
Epoch 10/10 | Loss: 0.066874 | Time: 1.31s
Training finished.
Model saved to Drive: /content/drive/MyDrive/Chessmate_Project/best_chess_model.pth
```

---

### Cell 18
```python

# ১. ONNX এক্সপোর্ট
model.eval()
dummy_input = torch.randn(1, 12, 8, 8).to(device)
onnx_path = os.path.join(project_folder, 'chess_model.onnx')

torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=12,
    do_constant_folding=True,
    input_names=['board_state'],
    output_names=['evaluation'],
    dynamic_axes={'board_state': {0: 'batch_size'}, 'evaluation': {0: 'batch_size'}}
)
print(f"ONNX model saved to Drive: {onnx_path}")

# ২. Hugging Face এ আপলোড
from huggingface_hub import HfApi

# আপনার তথ্য
HF_TOKEN = "MY-HF-TOKEN"
HF_USERNAME = "Rafs-an09002"

# মডেল রিপোজিটরি (নতুন)
MODEL_REPO_ID = f"{HF_USERNAME}/chessmate-model"

api = HfApi(token=HF_TOKEN)

print("\nUploading ONNX Model to Hugging Face...")

try:
    # মডেল রিপো তৈরি
    api.create_repo(repo_id=MODEL_REPO_ID, repo_type="model", exist_ok=True)

    # ফাইল আপলোড
    api.upload_file(
        path_or_fileobj=onnx_path,
        path_in_repo="chess_model.onnx",
        repo_id=MODEL_REPO_ID,
        repo_type="model"
    )
    print("🎉 ONNX Model Upload Complete!")
    print(f"Model URL: https://huggingface.co/models/{MODEL_REPO_ID}/resolve/main/chess_model.onnx")

except Exception as e:
    print(f"❌ Error uploading model: {e}")

```


Output:



```text
/tmp/ipython-input-844081963.py:6: UserWarning: # 'dynamic_axes' is not recommended when dynamo=True, and may lead to 'torch._dynamo.exc.UserError: Constraints violated.' Supply the 'dynamic_shapes' argument instead if export is unsuccessful.
  torch.onnx.export(
W1210 08:03:56.132000 256 torch/onnx/_internal/exporter/_compat.py:114] Setting ONNX exporter to use operator set version 18 because the requested opset_version 12 is a lower version than we have implementations for. Automatic version conversion will be performed, which may not be successful at converting to the requested version. If version conversion is unsuccessful, the opset version of the exported model will be kept at 18. Please consider setting opset_version >=18 to leverage latest ONNX features
[torch.onnx] Obtain model graph for `ChessNet([...]` with `torch.export.export(..., strict=False)`...
[torch.onnx] Obtain model graph for `ChessNet([...]` with `torch.export.export(..., strict=False)`... ✅
[torch.onnx] Run decomposition...
WARNING:onnxscript.version_converter:The model version conversion is not supported by the onnxscript version converter and fallback is enabled. The model will be converted using the onnx C API (target version: 12).
WARNING:onnxscript.version_converter:Failed to convert the model to the target version 12 using the ONNX C API. The model was not modified
Traceback (most recent call last):
  File "/usr/local/lib/python3.12/dist-packages/onnxscript/version_converter/__init__.py", line 127, in call
    converted_proto = _c_api_utils.call_onnx_api(
                      ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnxscript/version_converter/_c_api_utils.py", line 65, in call_onnx_api
    result = func(proto)
             ^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnxscript/version_converter/__init__.py", line 122, in _partial_convert_version
    return onnx.version_converter.convert_version(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.12/dist-packages/onnx/version_converter.py", line 39, in convert_version
    converted_model_str = C.convert_version(model_str, target_version)
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
RuntimeError: /github/workspace/onnx/version_converter/BaseConverter.h:68: adapter_lookup: Assertion `false` failed: No Adapter From Version $16 for Identity
[torch.onnx] Run decomposition... ✅
[torch.onnx] Translate the graph into ONNX...
[torch.onnx] Translate the graph into ONNX... ✅
Applied 4 of general pattern rewrite rules.
ONNX model saved to Drive: /content/drive/MyDrive/Chessmate_Project/chess_model.onnx

Uploading ONNX Model to Hugging Face...
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
Processing Files (1 / 1)      : 100%
 17.1kB / 17.1kB, 42.7kB/s  
New Data Upload               : 100%
 17.1kB / 17.1kB, 42.7kB/s  
  ..._Project/chess_model.onnx: 100%
 17.1kB / 17.1kB            
🎉 ONNX Model Upload Complete!
Model URL: https://huggingface.co/models/Rafs-an09002/chessmate-model/resolve/main/chess_model.onnx
```

---

### Cell 19
```python

# ১. শুধু মিসিং লাইব্রেরিটি ইন্সটল
!pip install onnx

import torch
import torch.nn as nn
import os
from google.colab import files

print(f"✅ Active PyTorch: {torch.__version__}")

# ২. মডেল আর্কিটেকচার
class ChessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layer = nn.Sequential(
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.fc_layer = nn.Sequential(
            nn.Linear(128 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)
        return torch.tanh(x)

# ৩. মডেল ইনিশিয়ালাইজেশন
model = ChessNet()
model.eval()

# ৪. এক্সপোর্ট (যেহেতু ভার্সন ঠিক আছে, তাই এখন স্ট্যাবল সেটিংস কাজ করবে)
print("Creating ONNX model...")
dummy_input = torch.randn(1, 12, 8, 8)
output_file = 'chess_model.onnx'

try:
    torch.onnx.export(
        model,
        dummy_input,
        output_file,
        export_params=True,
        opset_version=14,          # Web-standard
        do_constant_folding=True,
        input_names=['board_state'],
        output_names=['evaluation'],
        dynamic_axes=None
    )

    # ৫. সাইজ চেক এবং ডাউনলোড
    file_size = os.path.getsize(output_file) / (1024 * 1024)

    if file_size > 5:
        print(f"✅ SUCCESS! Model Created. Size: {file_size:.2f} MB")
        print("Downloading...")
        files.download(output_file)
    else:
        print(f"❌ Failed: File is too small ({file_size:.2f} MB).")

except Exception as e:
    print(f"❌ Export Failed: {e}")
```


Output:



```text
.Collecting onnx
  Using cached onnx-1.20.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl.metadata (8.4 kB)
Requirement already satisfied: numpy>=1.23.2 in /usr/local/lib/python3.12/dist-packages (from onnx) (2.0.2)
Requirement already satisfied: protobuf>=4.25.1 in /usr/local/lib/python3.12/dist-packages (from onnx) (5.29.5)
Requirement already satisfied: typing_extensions>=4.7.1 in /usr/local/lib/python3.12/dist-packages (from onnx) (4.15.0)
Requirement already satisfied: ml_dtypes>=0.5.0 in /usr/local/lib/python3.12/dist-packages (from onnx) (0.5.4)
Using cached onnx-1.20.0-cp312-abi3-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl (18.1 MB)
Installing collected packages: onnx
Successfully installed onnx-1.20.0
✅ Active PyTorch: 2.5.1+cu124
Creating ONNX model...
✅ SUCCESS! Model Created. Size: 16.50 MB
Downloading...
```

---
