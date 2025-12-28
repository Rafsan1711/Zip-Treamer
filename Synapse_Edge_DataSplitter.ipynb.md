## Notebook Name: Chessmate Data Prep.ipynb

---

## 1. Introduction
eta amar  GambitFlow/synapse-edge ei model er dataset 4 vag korar  notebook. 

---

## 2. Code Implementation

### cell 1
```python
# Cell 1: Environment Setup & Master Database Download
# Purpose: Prepare the environment to split the 5.5M+ dataset into 4 shards.

import os
import sqlite3
import shutil
from huggingface_hub import hf_hub_download, HfApi

# ১. কনফিগারেশন
HF_ORG = "GambitFlow"
REPO_ID = f"{HF_ORG}/Synapse-Edge-Data"
MASTER_FILENAME = "synapse_training_final.db"

# লোকাল কাজের ডিরেক্টরি
WORKING_DIR = "/content/splitting_lab"
os.makedirs(WORKING_DIR, exist_ok=True)

print(f"🚀 Starting Data Sharding Process for {REPO_ID}...")

# ২. মেইন ডেটাবেস ডাউনলোড করা
try:
    print(f"⏳ Downloading Master Database ({MASTER_FILENAME})...")
    master_db_path = hf_hub_download(
        repo_id=REPO_ID,
        filename=MASTER_FILENAME,
        repo_type="dataset",
        local_dir=WORKING_DIR
    )
    print(f"✅ Master DB Ready at: {master_db_path}")
    print(f"⚖️  Size: {os.path.getsize(master_db_path) / (1024**2):.2f} MB")

except Exception as e:
    print(f"❌ Error downloading master database: {e}")

# ৩. রো (Row) সংখ্যা চেক করা
conn = sqlite3.connect(master_db_path)
cursor = conn.cursor()
cursor.execute("SELECT COUNT(*) FROM training_data")
total_rows = cursor.fetchone()[0]
conn.close()

print(f"\n📊 Total Positions in Master DB: {total_rows:,}")
print(f"🎯 Target: 4 shards of ~{total_rows // 4:,} positions each.")
```
Output:

```text
🚀 Starting Data Sharding Process for GambitFlow/Synapse-Edge-Data...
⏳ Downloading Master Database (synapse_training_final.db)...
/usr/local/lib/python3.12/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
synapse_training_final.db: 100%
 1.12G/1.12G [00:10<00:00, 324MB/s]
✅ Master DB Ready at: /content/splitting_lab/synapse_training_final.db
⚖️  Size: 1071.69 MB

📊 Total Positions in Master DB: 5,551,558
🎯 Target: 4 shards of ~1,387,889 positions each.
```


---


### cell 2
```python
# Cell 2: Precise Data Sharding logic
# Technique: SQL ATTACH & LIMIT/OFFSET for O(1) memory efficiency

import sqlite3
import os
from tqdm import tqdm

def create_shard(shard_id, start_index, limit_count, master_path):
    shard_name = f"synapse_shard_{shard_id}.db"
    shard_path = os.path.join(WORKING_DIR, shard_name)

    # ১. নতুন শার্ড ডাটাবেস তৈরি
    if os.path.exists(shard_path): os.remove(shard_path)
    shard_conn = sqlite3.connect(shard_path)
    shard_cursor = shard_conn.cursor()

    # ২. স্কিমা (Schema) তৈরি করা - মেইন ডিবি এর সাথে হুবহু মিল রেখে
    shard_cursor.execute('''
        CREATE TABLE training_data (
            fen TEXT PRIMARY KEY,
            position_stats TEXT,
            best_move TEXT,
            is_tactical INTEGER,
            difficulty TEXT
        )
    ''')

    # ৩. স্পিড অপ্টিমাইজেশন
    shard_cursor.execute('PRAGMA synchronous = OFF')
    shard_cursor.execute('PRAGMA journal_mode = MEMORY')

    # ৪. মেইন ডাটাবেসকে এই কানেকশনে সংযুক্ত (ATTACH) করা
    shard_cursor.execute(f"ATTACH DATABASE '{master_path}' AS master_db")

    # ৫. নির্দিষ্ট রেঞ্জের ডেটা কপি করা (The actual splitting)
    print(f"📦 Extracting Shard {shard_id}: Positions {start_index:,} to {start_index + limit_count:,}")
    shard_cursor.execute(f'''
        INSERT INTO training_data
        SELECT * FROM master_db.training_data
        LIMIT {limit_count} OFFSET {start_index}
    ''')

    shard_conn.commit()

    # ৬. ভেরিফিকেশন
    shard_cursor.execute("SELECT COUNT(*) FROM training_data")
    count = shard_cursor.fetchone()[0]

    shard_cursor.execute("DETACH DATABASE master_db")
    shard_conn.close()

    return shard_name, count

# --- স্লাইসিং ক্যালকুলেশন ---
total_rows = 5551558
rows_per_shard = total_rows // 4  # ১,৩৮৭,৮৮৯ টি করে

print(f"🏗️  Starting Sharding Engine...")
shards_info = []

for i in range(1, 5):
    start = (i - 1) * rows_per_shard
    # শেষ শার্ডের জন্য অবশিষ্ট সবটুকু (বাকি থাকা ০.৫ বা ১-২ রো) নিয়ে নিবে
    limit = rows_per_shard if i < 4 else (total_rows - start)

    name, count = create_shard(i, start, limit, master_db_path)
    shards_info.append((name, count))

print("\n" + "="*50)
print("✅ SHARDING COMPLETE!")
print("="*50)
for name, count in shards_info:
    size = os.path.getsize(os.path.join(WORKING_DIR, name)) / (1024**2)
    print(f"📄 {name} | Positions: {count:,} | Size: {size:.2f} MB")
```

Output:

```text
🏗️  Starting Sharding Engine...
📦 Extracting Shard 1: Positions 0 to 1,387,889
📦 Extracting Shard 2: Positions 1,387,889 to 2,775,778
📦 Extracting Shard 3: Positions 2,775,778 to 4,163,667
📦 Extracting Shard 4: Positions 4,163,667 to 5,551,558

==================================================
✅ SHARDING COMPLETE!
==================================================
📄 synapse_shard_1.db | Positions: 1,387,889 | Size: 320.71 MB
📄 synapse_shard_2.db | Positions: 1,387,889 | Size: 314.58 MB
📄 synapse_shard_3.db | Positions: 1,387,889 | Size: 186.60 MB
📄 synapse_shard_4.db | Positions: 1,387,891 | Size: 186.43 MB
```
---

### cell 3
```python
# Cell 3: Uploading the 4 Data Shards to Hugging Face
# Purpose: Finalizing the distributed dataset for a 4-account training setup.

from huggingface_hub import HfApi
import os
import time

# --- ১. ক্রেডেনশিয়ালস এবং কনফিগারেশন ---
# আপনার Hugging Face 'Write' টোকেনটি এখানে বসান
HF_TOKEN = "HF_TOKEN"

HF_ORG = "GambitFlow"
REPO_ID = f"{HF_ORG}/Synapse-Edge-Data"

# আমাদের লোকাল ডিরেক্টরি যেখানে শার্ডগুলো আছে
WORKING_DIR = "/content/splitting_lab"

api = HfApi(token=HF_TOKEN)

print(f"🚀 Initializing Bulk Upload to: {REPO_ID}")

# ৪টি শার্ডের তালিকা
shards = [
    "synapse_shard_1.db",
    "synapse_shard_2.db",
    "synapse_shard_3.db",
    "synapse_shard_4.db"
]

def upload_all_shards():
    for shard_name in shards:
        local_path = os.path.join(WORKING_DIR, shard_name)

        if not os.path.exists(local_path):
            print(f"⚠️  Warning: {shard_name} not found locally. Skipping...")
            continue

        file_size = os.path.getsize(local_path) / (1024**2)
        print(f"\n📦 Preparing: {shard_name} ({file_size:.2f} MB)")

        try:
            # ফাইল আপলোড (Direct Stream)
            # আমরা ফাইলগুলো 'shards/' নামক একটি ফোল্ডারের ভেতরে রাখব যাতে রিপোজিটরি পরিষ্কার থাকে
            start_time = time.time()
            api.upload_file(
                path_or_fileobj=local_path,
                path_in_repo=f"shards/{shard_name}",
                repo_id=REPO_ID,
                repo_type="dataset"
            )
            end_time = time.time()
            print(f"✅ Uploaded {shard_name} in {end_time - start_time:.2f} seconds.")

        except Exception as e:
            print(f"❌ Failed to upload {shard_name}: {e}")

# ২. আপলোড শুরু
try:
    # রিপোজিটরি চেক (সেফটি হিসেবে)
    api.create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    upload_all_shards()

    print("\n" + "="*60)
    print("🎉 MISSION ACCOMPLISHED: ALL SHARDS ARE LIVE!")
    print("="*60)
    print(f"🔗 Check Shards Here: https://huggingface.co/datasets/{REPO_ID}/tree/main/shards")
    print("="*60)

except Exception as e:
    print(f"❌ Critical Error: {e}")
```

Output:

```text
🚀 Initializing Bulk Upload to: GambitFlow/Synapse-Edge-Data

📦 Preparing: synapse_shard_1.db (320.71 MB)
Processing Files (1 / 1)      : 100%
  336MB /  336MB, 70.1MB/s  
New Data Upload               : 100%
  336MB /  336MB, 70.1MB/s  
  ...ng_lab/synapse_shard_1.db: 100%
  336MB /  336MB            
✅ Uploaded synapse_shard_1.db in 7.65 seconds.

📦 Preparing: synapse_shard_2.db (314.58 MB)
Processing Files (1 / 1)      : 100%
  330MB /  330MB, 56.9MB/s  
New Data Upload               : 100%
  330MB /  330MB, 56.9MB/s  
  ...ng_lab/synapse_shard_2.db: 100%
  330MB /  330MB            
✅ Uploaded synapse_shard_2.db in 7.93 seconds.

📦 Preparing: synapse_shard_3.db (186.60 MB)
Processing Files (1 / 1)      : 100%
  196MB /  196MB, 48.9MB/s  
New Data Upload               : 100%
  196MB /  196MB, 48.9MB/s  
  ...ng_lab/synapse_shard_3.db: 100%
  196MB /  196MB            
✅ Uploaded synapse_shard_3.db in 6.72 seconds.

📦 Preparing: synapse_shard_4.db (186.43 MB)
Processing Files (1 / 1)      : 100%
  195MB /  195MB, 37.6MB/s  
New Data Upload               : 100%
  195MB /  195MB, 37.6MB/s  
  ...ng_lab/synapse_shard_4.db: 100%
  195MB /  195MB            
✅ Uploaded synapse_shard_4.db in 6.99 seconds.

============================================================
🎉 MISSION ACCOMPLISHED: ALL SHARDS ARE LIVE!
============================================================
🔗 Check Shards Here: https://huggingface.co/datasets/GambitFlow/Synapse-Edge-Data/tree/main/shards
============================================================
```
---
