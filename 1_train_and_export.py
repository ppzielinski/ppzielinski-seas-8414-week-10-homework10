"""
1_train_and_export.py

End-to-end script that:
1) Generates a toy DGA training dataset (dga_dataset_train.csv)
2) Trains an H2O AutoML model on it
3) Exports the leader model as a MOJO into ./model

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
# Data generation (from lab code)
# -----------------------------
def get_entropy(s: str) -> float:
    p, lns = {}, float(len(s))
    for c in s:
        p[c] = p.get(c, 0) + 1
    return -sum(count/lns * math.log(count/lns, 2) for count in p.values())


def generate_dataset(csv_path: str) -> None:
    """Generate dga_dataset_train.csv if it doesn't already exist."""
    if os.path.exists(csv_path):
        print(f"[data] {csv_path} already exists. Skipping generation.")
        return

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
    """Train H2O AutoML on the dataset and export leader MOJO to model_dir.
    Returns the path to the saved MOJO.
    """
    print("[h2o] Initializing H2O...")
    h2o.init()

    print(f"[h2o] Importing training data from {csv_path}...")
    train = h2o.import_file(csv_path)

    x = ['length', 'entropy']  # Features, per lab
    y = 'class'                # Target
    train[y] = train[y].asfactor()

    print("[automl] Running H2O AutoML (max_models=20, max_runtime_secs=120)...")
    aml = H2OAutoML(max_models=20, max_runtime_secs=120, seed=1)
    aml.train(x=x, y=y, training_frame=train)

    print("[automl] Leaderboard (top rows):")
    print(aml.leaderboard.head(rows=10))

    best_model = aml.leader
    Path(model_dir).mkdir(parents=True, exist_ok=True)

    print(f"[mojo] Downloading leader MOJO to {model_dir}...")
    mojo_path = best_model.download_mojo(path=model_dir)
    print(f"[mojo] Production-ready model saved to: {mojo_path}")

    print("[h2o] Shutting down H2O...")
    h2o.shutdown(prompt=False)

    return mojo_path


def main():
    csv_path = "dga_dataset_train.csv"
    model_dir = "./model"  # per homework requirement

    generate_dataset(csv_path)
    mojo_path = train_and_export(csv_path, model_dir=model_dir)

    print("\nDone.")
    print(f"Dataset: {csv_path}")
    print(f"MOJO:    {mojo_path}")


if __name__ == "__main__":
    main()