#!/usr/bin/env python3
"""
1_train_and_export.py (updated)

Runs the end-to-end Part 1 pipeline:
1) Generate dga_dataset_train.csv (always regenerates for reproducibility)
2) Train H2O AutoML on (length, entropy) features
3) Export the leader model as a MOJO saved EXACTLY as ./model/DGA_Leader.zip

Requirements:
- Python 3.x
- h2o (pip install h2o)
- Java runtime (for H2O)
"""

import os
import math
import random
import csv
from pathlib import Path

import h2o
from h2o.automl import H2OAutoML


# -----------------------------
# Data generation (lab logic)
# -----------------------------
def get_entropy(s: str) -> float:
    p, lns = {}, float(len(s))
    for c in s:
        p[c] = p.get(c, 0) + 1
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())


def generate_dataset(csv_path: str) -> None:
    """Always (re)generate the training dataset."""
    header = ['domain', 'length', 'entropy', 'class']
    data = []

    # Legitimate domains
    legit_domains = ['google', 'facebook', 'amazon', 'github', 'wikipedia', 'microsoft']
    for _ in range(100):
        domain = random.choice(legit_domains) + ".com"
        data.append([domain, len(domain), get_entropy(domain), 'legit'])

    # DGA-like domains
    for _ in range(100):
        length = random.randint(15, 25)
        domain = ''.join(random.choice('abcdefghijklmnopqrstuvwxyz0123456789') for _ in range(length)) + ".com"
        data.append([domain, len(domain), get_entropy(domain), 'dga'])

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)

    print(f"[data] {csv_path} created successfully.")


# -----------------------------
# AutoML training + MOJO export
# -----------------------------
def train_and_export(csv_path: str, model_dir: str = "./model") -> str:
    """Train H2O AutoML and export the leader MOJO as ./model/DGA_Leader.zip."""
    print("[h2o] Initializing H2O...")
    h2o.init()

    print(f"[h2o] Importing training data from {csv_path}...")
    train = h2o.import_file(csv_path)

    x = ['length', 'entropy']  # Features from lab
    y = 'class'                # Target
    train[y] = train[y].asfactor()

    print("[automl] Running H2O AutoML (max_models=20, max_runtime_secs=120)...")
    aml = H2OAutoML(max_models=20, max_runtime_secs=120, seed=1)
    aml.train(x=x, y=y, training_frame=train)

    print("[automl] Leaderboard (top rows):")
    print(aml.leaderboard.head(rows=10))

    best_model = aml.leader
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    # Export MOJO to a temp filename, then copy to DGA_Leader.zip
    print(f"[mojo] Downloading leader MOJO to {model_dir}...")
    mojo_temp = best_model.download_mojo(path=model_dir)

    # Normalize to required filename
    target_path = str(Path(model_dir) / "DGA_Leader.zip")
    try:
        # Overwrite if exists
        import shutil
        shutil.copyfile(mojo_temp, target_path)
        print(f"[mojo] Production-ready model saved to: {target_path}")
    except Exception as e:
        print(f"[mojo] Could not copy MOJO to {target_path}: {e}")
        target_path = mojo_temp
        print(f"[mojo] Using original MOJO path instead: {target_path}")

    # Graceful shutdown
    try:
        h2o.cluster().shutdown()
    except Exception:
        pass

    return target_path


def main():
    csv_path = "dga_dataset_train.csv"
    model_dir = "./model"

    generate_dataset(csv_path)               # always regenerate
    mojo_path = train_and_export(csv_path, model_dir=model_dir)

    print("\nDone.")
    print(f"Dataset: {csv_path}")
    print(f"MOJO:    {mojo_path}")


if __name__ == "__main__":
    main()
