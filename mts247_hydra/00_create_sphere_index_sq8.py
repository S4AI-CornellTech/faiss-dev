import os
import math
import numpy as np
import faiss
from datasets import load_dataset
from tqdm import tqdm

# ==============================
# Config
# ==============================
TARGET_COUNT = 600_000_000
TRAIN_SAMPLE_SIZE = 1_000_000
BATCH_SIZE = 100_000 
SAVE_POINTS = [300_000_000, 500_000_000, 600_000_000]
OUTPUT_DIR = "/data/indices"

def build_600m_index():
    dataset_name = "mohdumar/SPHERE_899M"
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"\n--- Building Index from {dataset_name} ---")

    # Use streaming=True to handle massive datasets without a local download crash
    ds = load_dataset(dataset_name, split="train", streaming=True)

    # 1. Training Phase
    print(f"Collecting {TRAIN_SAMPLE_SIZE:,} vectors for training...")
    train_vectors = []
    for item in tqdm(ds.take(TRAIN_SAMPLE_SIZE), total=TRAIN_SAMPLE_SIZE, desc="Training Set"):
        train_vectors.append(item["vector"])
    
    train_vectors = np.array(train_vectors, dtype=np.float32)
    dim = train_vectors.shape[1]
    # Adjust nlists for the larger 600m target
    nlists = int(math.sqrt(TARGET_COUNT))

    print(f"Training IVF-SQ8 (nlists={nlists})...")
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFScalarQuantizer(
        quantizer, dim, nlists, faiss.ScalarQuantizer.QT_8bit, faiss.METRIC_INNER_PRODUCT
    )
    index.train(train_vectors)
    del train_vectors # Immediate RAM cleanup

    # 2. Fast Indexing Phase
    print(f"Indexing up to {TARGET_COUNT:,} vectors...")
    pbar = tqdm(total=TARGET_COUNT, desc="Indexing", unit="vec")
    
    count = 0
    saved_milestones = set()

    for batch in ds.iter(batch_size=BATCH_SIZE):
        vecs = np.array(batch["vector"], dtype=np.float32)
        index.add(vecs)
        
        count += len(vecs)
        pbar.update(len(vecs))
        
        # Check for save points
        for milestone in SAVE_POINTS:
            if count >= milestone and milestone not in saved_milestones:
                print(f"\nReached {milestone:,} milestone. Saving index...")
                save_name = f"sphere_{milestone // 1_000_000}m_ivf_sq8.faiss"
                save_path = os.path.join(OUTPUT_DIR, save_name)
                faiss.write_index(index, save_path)
                saved_milestones.add(milestone)

        if count >= TARGET_COUNT:
            break

    print(f"Build complete. Total vectors indexed: {index.ntotal:,}")

if __name__ == "__main__":
    build_600m_index()