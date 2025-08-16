#!/usr/bin/env python3
"""
2_analyze_domain.py

CLI that:
- Loads the production MOJO from ./model/DGA_Leader.zip (default, override with --model)
- Engineers features (length, entropy) for --domain
- Predicts legit|dga
- If dga, builds a SHAP explanation for the single instance and passes a structured
  summary to Google Gemini via your 4_generate_prescriptive_playbook.py helper.

Usage:
  python 2_analyze_domain.py --domain google.com
  python 2_analyze_domain.py --domain kq3v9z7j1x5f8g2h.info --model ./model/DGA_Leader.zip
"""

import argparse
import asyncio
import math
import os
import random
import sys
import importlib.util

import h2o
import numpy as np
import pandas as pd
import shap

# Load .env so you don't need --api-key
from dotenv import load_dotenv
load_dotenv()
GENAI_API_KEY = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")

# --- dynamic import of 4_generate_prescriptive_playbook.py (filename starts with digit) ---
_mod_path = os.path.join(os.path.dirname(__file__), "4_generate_prescriptive_playbook.py")
spec = importlib.util.spec_from_file_location("gen_playbook_mod", _mod_path)
if spec and spec.loader:
    gen_playbook_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen_playbook_mod)  # type: ignore[attr-defined]
    generate_playbook = getattr(gen_playbook_mod, "generate_playbook", None)
else:
    generate_playbook = None
# -----------------------------------------------------------------------------


# -----------------------------
# Feature engineering (lab logic)
# -----------------------------
def get_entropy(s: str) -> float:
    p, lns = {}, float(len(s))
    for c in s:
        p[c] = p.get(c, 0) + 1
    return -sum((cnt/lns) * math.log((cnt/lns), 2) for cnt in p.values()) if lns > 0 else 0.0


def sanitize_domain(d: str) -> str:
    d = d.strip()
    d = d.replace("http://", "").replace("https://", "").replace("ftp://", "")
    d = d.split("/")[0].split("?")[0]
    return d


def build_features(domain: str) -> pd.DataFrame:
    d = sanitize_domain(domain)
    return pd.DataFrame([{
        "domain": d,
        "length": len(d),
        "entropy": get_entropy(d),
    }])


# -----------------------------
# Robust probability-column handling
# -----------------------------
def _calibrate_dga_prob_column(model, h2o_frame_cols=("length", "entropy")) -> str:
    """
    Pick the column that represents P(dga), robust to different MOJO outputs:
    - Prefer a column literally named 'dga' (case-insensitive).
    - Else, prefer 'p1' if present, then 'p0'.
    - Else, if there are exactly two probability columns, choose the one larger on a DGA-like row.
    - Else, fall back to the first non-'predict' column.
    """
    import pandas as _pd
    import h2o as _h2o

    # Score two rows to inspect columns
    legit = {"length": len("google.com"), "entropy": get_entropy("google.com")}
    dga_like = "".join(random.choice("abcdefghijklmnopqrstuvwxyz0123456789") for _ in range(20)) + ".com"
    dga = {"length": len(dga_like), "entropy": get_entropy(dga_like)}
    fr = _h2o.H2OFrame(_pd.DataFrame([legit, dga], columns=list(h2o_frame_cols)))
    preds = model.predict(fr).as_data_frame()

    # Identify probability columns (exclude 'predict')
    prob_cols = [c for c in preds.columns if c.lower() != "predict"]
    if not prob_cols:
        return "p1"  # best-effort fallback

    # 1) 'dga' column by name
    for c in prob_cols:
        if c.lower() == "dga":
            return c

    # 2) Standard H2O names
    if "p1" in prob_cols:
        return "p1"
    if "p0" in prob_cols:
        return "p0"

    # 3) Two columns: pick the larger on DGA-like row
    if len(prob_cols) == 2:
        dga_row = preds.iloc[1]
        c0, c1 = prob_cols
        return c1 if dga_row[c1] >= dga_row[c0] else c0

    # 4) First non-predict column
    return prob_cols[0]


def make_predict_proba_fn(model, dga_prob_col: str):
    """
    Return f(X) for SHAP KernelExplainer that outputs P(dga).
    Works with either ['dga','legit'] or ['p0','p1'] style outputs.
    """
    def f(X):
        import pandas as _pd
        import numpy as _np
        import h2o as _h2o

        if isinstance(X, _np.ndarray):
            df = _pd.DataFrame(X, columns=["length", "entropy"])
        elif isinstance(X, _pd.DataFrame):
            df = X[["length", "entropy"]].copy()
        else:
            df = _pd.DataFrame(X, columns=["length", "entropy"])
        hf = _h2o.H2OFrame(df)
        preds = model.predict(hf).as_data_frame()

        # If our chosen column is missing, try to recover gracefully
        cols_lower = {c.lower(): c for c in preds.columns}
        if dga_prob_col in preds.columns:
            col = dga_prob_col
        elif "dga" in cols_lower:
            col = cols_lower["dga"]
        elif "p1" in preds.columns:
            col = "p1"
        else:
            prob_cols = [c for c in preds.columns if c.lower() != "predict"]
            if len(prob_cols) == 2:
                # choose the column with higher mean as a crude fallback
                col = prob_cols[int(preds[prob_cols[1]].mean() > preds[prob_cols[0]].mean())]
            else:
                col = prob_cols[0]
        return preds[col].values
    return f


def summarize_shap(domain: str, instance_df: pd.DataFrame, shap_values, expected_value, pred_prob: float) -> str:
    feature_names = ["length", "entropy"]
    if isinstance(shap_values, np.ndarray):
        vals = shap_values.tolist()
    else:
        vals = list(shap_values)
    pairs = list(zip(feature_names, instance_df[feature_names].iloc[0].tolist(), vals))
    pairs_sorted = sorted(pairs, key=lambda t: abs(t[2]), reverse=True)

    bullets = []
    for fname, fval, s in pairs_sorted:
        direction = "pushed towards 'dga'" if s > 0 else "pushed towards 'legit'"
        bullets.append(f"  - {fname} = {fval} (SHAP {s:+.3f}, {direction})")

    text = (
        f"- Alert: Potential DGA domain detected.\n"
        f"- Domain: '{instance_df['domain'].iloc[0]}'\n"
        f"- AI Model Explanation (from SHAP): The model flagged this domain with {pred_prob*100:.1f}% confidence for 'dga'.\n"
        f"  The classification was primarily driven by:\n" + "\n".join(bullets)
    )
    return text


# -----------------------------
# Main analysis
# -----------------------------
def analyze_domain(domain: str, model_path: str) -> int:
    # 1) Start H2O & load MOJO
    h2o.init()
    if not os.path.exists(model_path):
        print(f"[error] MOJO not found at: {model_path}", file=sys.stderr)
        return 2
    print(f"[model] Importing MOJO from: {model_path}")
    model = h2o.import_mojo(model_path)

    # 2) Features
    inst = build_features(domain)
    print("[features] Single-instance features:")
    print(inst[["length", "entropy"]])

    # 3) Predict
    hf = h2o.H2OFrame(inst[["length", "entropy"]])
    preds = model.predict(hf).as_data_frame()
    print("\n[predict] Raw prediction output:")
    print(preds)

    # normalize column names for readability
    cols_lower = {c.lower(): c for c in preds.columns}
    pred_col = cols_lower.get("predict", "predict")
    predicted = str(preds.loc[0, pred_col]).lower()

    if predicted == "dga":
        # 4) SHAP explanation on P(dga)
        dga_prob_col = _calibrate_dga_prob_column(model)
        f = make_predict_proba_fn(model, dga_prob_col)

        # background: prefer your training csv if present, else synthetic
        if os.path.exists("dga_dataset_train.csv"):
            try:
                bg = pd.read_csv("dga_dataset_train.csv")[["length", "entropy"]].sample(
                    n=50, random_state=42
                )
            except Exception:
                bg = pd.DataFrame({
                    "length": np.random.default_rng(42).integers(5, 30, size=50),
                    "entropy": np.random.default_rng(43).uniform(1.0, 5.0, size=50),
                })
        else:
            bg = pd.DataFrame({
                "length": np.random.default_rng(42).integers(5, 30, size=50),
                "entropy": np.random.default_rng(43).uniform(1.0, 5.0, size=50),
            })

        explainer = shap.KernelExplainer(f, bg)
        x = inst[["length", "entropy"]]
        shap_vals = explainer.shap_values(x, nsamples=100)
        svals = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
        if isinstance(svals, np.ndarray) and svals.ndim > 1:
            svals = svals[0]
        pdga = float(f(x)[0])

        xai_findings = summarize_shap(domain, inst, svals, getattr(explainer, "expected_value", 0.0), pdga)
        print("\n--- XAI Findings (auto-generated) ---")
        print(xai_findings)

        # 5) GenAI playbook
        if generate_playbook is None:
            print("\n[warn] Skipping GenAI playbook: generate_playbook not found (missing 4_generate_prescriptive_playbook.py).")
        elif not GENAI_API_KEY:
            print("\n[warn] Skipping GenAI playbook: missing GEMINI_API_KEY / GOOGLE_API_KEY in .env.")
        else:
            print("\n--- AI-Generated Playbook ---")
            playbook = asyncio.run(generate_playbook(xai_findings, GENAI_API_KEY))
            print(playbook)

    else:
        print(f"\n[result] Predicted class: LEGIT")
        print("[info] Playbook generation is only triggered for DGA predictions.")

    # Shutdown
    try:
        h2o.cluster().shutdown()
    except Exception:
        pass
    return 0


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Analyze a domain: predict, explain (SHAP), and prescribe (GenAI).")
    ap.add_argument("--domain", required=True, help="Domain to analyze (e.g., google.com)")
    ap.add_argument("--model", default="./model/DGA_Leader.zip",
                    help="Path to MOJO zip (default: ./model/DGA_Leader.zip)")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    try:
        return analyze_domain(args.domain, args.model)
    except KeyboardInterrupt:
        print("\n[abort] Interrupted by user.", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
