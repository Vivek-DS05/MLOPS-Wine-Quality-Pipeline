import pandas as pd
import mlflow
import yaml
import pickle
import json
import os
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

def evaluate():
    # Load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    mlflow_params = params["mlflow"]

    # Load data
    test_df = pd.read_csv("data/test.csv")
    X_test = test_df.drop("quality_binary", axis=1)
    y_test = test_df["quality_binary"]

    # Load model
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)

    # Load run_id
    with open("models/run_id.txt") as f:
        run_id = f.read().strip()

    # Predict
    preds = model.predict(X_test)

    # Metrics
    metrics = {
        "test_accuracy": round(accuracy_score(y_test, preds), 4),
        "test_f1": round(f1_score(y_test, preds), 4),
        "test_precision": round(precision_score(y_test, preds), 4),
        "test_recall": round(recall_score(y_test, preds), 4),
    }

    # Log to MLflow (resume same run)
    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metrics(metrics)

    # Save metrics for DVC
    os.makedirs("metrics", exist_ok=True)
    with open("metrics/scores.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("✅ Evaluation complete!")
    for k, v in metrics.items():
        print(f"   {k}: {v}")

if __name__ == "__main__":
    evaluate()