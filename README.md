# DGA Domain Detection Pipeline

## Project Goal
This project is designed to detect **Domain Generation Algorithm (DGA)** domains using a three-stage machine learning pipeline. 
The goal is to distinguish between **malicious DGA-generated domains** and **legitimate domains** in order to support 
cybersecurity triage, analysis, and automated playbook generation.

---

## Three-Stage Architecture

1. **Model Training & Export (`1_train_and_export.py`)**
   - Trains a machine learning classifier on the provided dataset (`dga_dataset_train.csv`).
   - Exports the trained model for use in later stages.

2. **Domain Analysis (`2_analyze_domain.py`)**
   - Loads the trained model and analyzes a given domain name.
   - Outputs whether the domain is **DGA-generated (malicious)** or **legitimate**.

3. **Prescriptive Playbook Generation (`4_generate_prescriptive_playbook.py`)**
   - Builds an automated **playbook of actions** based on the classification result.
   - Example: If a domain is flagged as malicious, it may suggest blocking the domain, isolating the host, or alerting the SOC.

---

## Installation

Clone the repository and install the dependencies:

```bash
git clone <repository-url>
cd project-directory
python -m venv .venv
source .venv/bin/activate   # On Linux/Mac
.venv\Scripts\activate    # On Windows

pip install -r requirements.txt
```

---

## Usage

### 1. Train the Model
```bash
python 1_train_and_export.py --input dga_dataset_train.csv --output model.pkl
```

### 2. Analyze a Domain
```bash
python 2_analyze_domain.py --model model.pkl --domain google.com
```
**Example Output:**
```
Domain: google.com
Prediction: Legitimate
```

```bash
python 2_analyze_domain.py --model model.pkl --domain asdfqwer.biz
```
**Example Output:**
```
Domain: asdfqwer.biz
Prediction: Malicious (DGA)
```

### 3. Generate a Prescriptive Playbook
```bash
python 4_generate_prescriptive_playbook.py --model model.pkl --domain asdfqwer.biz
```
**Example Output:**
```
Domain: asdfqwer.biz
Prediction: Malicious (DGA)
Recommended Actions:
- Block the domain at firewall/DNS level
- Isolate the infected host
- Alert SOC for further investigation
```

---

## Project Structure
```
.
├── 1_train_and_export.py          # Model training and export script
├── 2_analyze_domain.py            # Domain classification script
├── 4_generate_prescriptive_playbook.py # Automated playbook generator
├── dga_dataset_train.csv          # Training dataset
├── requirements.txt               # Python dependencies
├── README.md                      # Project documentation
└── TESTING.md                     # Manual testing guide
```

---

## Notes
- The pipeline can be extended to support more advanced feature extraction and different classifiers.
- Example playbooks can be integrated with SOAR tools for automated response.
