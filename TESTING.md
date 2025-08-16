# TESTING.md

## Manual Verification Steps

This document describes how to manually test the DGA Detection Pipeline with **one known legitimate domain** and **one known malicious (DGA) domain**.

---

### Prerequisites
- Python 3.11+
- Installed dependencies (`pip install -r requirements.txt`)
- Trained model (`model.pkl` generated using `1_train_and_export.py`)

---

## Example 1: Legitimate Domain (`google.com`)

Run the command:
```bash
python 2_analyze_domain.py --model model.pkl --domain google.com
```

**Expected Output:**
```
Domain: google.com
Prediction: Legitimate
```

---

## Example 2: Malicious DGA Domain (`asdfqwer.biz`)

Run the command:
```bash
python 2_analyze_domain.py --model model.pkl --domain asdfqwer.biz
```

**Expected Output:**
```
Domain: asdfqwer.biz
Prediction: Malicious (DGA)
```

---

## Playbook Generation Test

To verify automated recommendations, run:
```bash
python 4_generate_prescriptive_playbook.py --model model.pkl --domain asdfqwer.biz
```

**Expected Output:**
```
Domain: asdfqwer.biz
Prediction: Malicious (DGA)
Recommended Actions:
- Block the domain at firewall/DNS level
- Isolate the infected host
- Alert SOC for further investigation
```

---

## Conclusion
If both tests produce the expected output, the pipeline is functioning correctly. 
Legitimate domains should be classified as **Legitimate**, and suspicious/random DGA-like domains should be classified as **Malicious (DGA)**.
