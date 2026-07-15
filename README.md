# 🍷 Wine Quality MLOps Pipeline

A reproducible ML pipeline using **DVC**, **MLflow**, and **Docker**.

---

## 🏗️ Architecture

```
params.yaml
     │
     ▼
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Prepare   │────▶│    Train    │────▶│  Evaluate   │
│   (DVC)     │     │  (MLflow)   │     │  (MLflow)   │
└─────────────┘     └─────────────┘     └─────────────┘
     │                    │                    │
     ▼                    ▼                    ▼
data/train.csv      models/model.pkl    metrics/scores.json
data/test.csv       models/run_id.txt
```

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **DVC** | Data & pipeline versioning |
| **MLflow** | Experiment tracking |
| **Docker** | Containerization |
| **Scikit-learn** | ML model |
| **Python** | Core language |

---

## 🚀 How to Run

### Option A: Docker (Recommended)
```bash
docker-compose up --build
```
Then open **http://localhost:5000** for MLflow UI

### Option B: Local
```bash
# Install dependencies
pip install -r requirements.txt

# Start MLflow server (Terminal 1)
mlflow server --host 0.0.0.0 --port 5000

# Run pipeline (Terminal 2)
dvc init --no-scm
dvc repro
```

---

## 📊 Results

```
test_accuracy:  0.8125
test_f1:        0.8341
test_precision: 0.8203
test_recall:    0.8483
```

---

## 📁 Project Structure

```
wine-quality-predictor/
├── data/               # Data files (tracked by DVC)
├── models/             # Model files (tracked by DVC)
├── src/
│   ├── prepare_data.py # Data preparation stage
│   ├── train.py        # Training stage
│   └── evaluate.py     # Evaluation stage
├── dvc.yaml            # DVC pipeline definition
├── params.yaml         # Experiment parameters
├── Dockerfile          
├── docker-compose.yml  
└── requirements.txt    
```

---

## 🔄 Run Experiments

```bash
# Change params in params.yaml
# For example change n_estimators: 100 to 200

# Re-run pipeline
dvc repro

# Check metrics
dvc metrics show
```

---

## 📸 MLflow UI
After running, open http://localhost:5000 to see:
- All experiment runs
- Parameters used
- Metrics comparison
- Saved models