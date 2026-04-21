# 🛡️ Automated MITRE ATT&CK Mapping

> Mini Group Project — Data Driven Problem Solving (Minor 6)  
> Seamedu School of Pro-Expressionism | Batch 2023-26

---

## 📌 Problem Statement

Security analysts receive network alerts but lack context about **what attack stage** is occurring.  
This system automatically classifies a network log entry into one of five MITRE ATT&CK tactics using a trained XGBoost classifier.

---

## 🗺️ MITRE Tactic Mapping

| NSL-KDD Category | MITRE Tactic        | Tactic ID | Technique                           |
|-----------------|---------------------|-----------|-------------------------------------|
| Normal          | No Threat           | —         | Benign Traffic                      |
| DoS             | Impact              | TA0040    | T1499 Endpoint Denial of Service    |
| Probe           | Discovery           | TA0007    | T1046 Network Service Scanning      |
| R2L             | Initial Access      | TA0001    | T1190 Exploit Public-Facing App     |
| U2R             | Privilege Escalation| TA0004    | T1068 Exploitation for Priv. Esc.   |

---

## 🏗️ Project Structure

```
mitre_attack_mapper/
├── app.py                   # Streamlit application (main entry point)
├── train_model.py           # Standalone training script
├── requirements.txt         # Python dependencies
├── README.md
│
├── src/
│   ├── __init__.py
│   ├── preprocess.py        # Data loading, encoding, preparation
│   ├── model.py             # XGBoost training, prediction, download
│   └── mitre_mapping.py     # MITRE tactic definitions and mappings
│
├── data/
│   └── README.md            # Dataset auto-downloads at runtime
│
└── .streamlit/
    └── config.toml          # Dark theme and server settings
```

---

## ⚙️ Tech Stack

| Component | Technology |
|-----------|-----------|
| ML Model | XGBoost + Scikit-learn |
| Dataset | NSL-KDD (public benchmark) |
| Dashboard | Streamlit |
| Visualization | Plotly, Matplotlib |
| Cloud Deployment | Streamlit Cloud |

---

## 🚀 Running Locally

```bash
# 1. Clone the repository
git clone https://github.com/YOUR_USERNAME/mitre-attack-mapper.git
cd mitre-attack-mapper

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. (Optional) Pre-train and cache the model
python train_model.py

# 5. Run the Streamlit app
streamlit run app.py
```

---

## ☁️ Deploying to Streamlit Cloud

1. Push this repository to **GitHub**
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repository, branch `main`, and main file `app.py`
4. Click **Deploy** — done!

> The app downloads the NSL-KDD dataset and trains the model automatically on first run (~60 seconds).  
> Subsequent runs load the cached model instantly.

---

## 📊 Deliverables Checklist

- [x] **Problem Definition** — Clear statement, objectives, success metrics
- [x] **Data Collection** — NSL-KDD public dataset with quality validation
- [x] **Data Analysis** — EDA with class distribution, feature analysis, box plots
- [x] **Modeling** — XGBoost multi-class classifier with accuracy ≥ 98%
- [x] **Visualization** — Interactive Plotly dashboard, kill-chain diagram
- [x] **Cloud Integration** — Deployed on Streamlit Cloud
- [x] **Ethical Considerations** — NSL-KDD is a public benchmark; no PII used

---

## 👥 Group Members

| Name | Roll No | Contribution |
|------|---------|--------------|
|      |         |              |
|      |         |              |
|      |         |              |

---

## 📚 References

- MITRE ATT&CK Framework: https://attack.mitre.org
- NSL-KDD Dataset: https://github.com/defcom17/NSL_KDD
- Tavallaee et al., "A Detailed Analysis of the KDD CUP 99 Data Set" (2009)
