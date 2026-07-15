import pandas as pd
import mlflow
import mlflow.sklearn
import yaml
import os
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

def train():
    # Load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)

    train_params = params["train"]
    mlflow_params = params["mlflow"]

    # Load data
    train_df = pd.read_csv("data/train.csv")
    X_train = train_df.drop("quality_binary", axis=1)
    y_train = train_df["quality_binary"]

    # Setup MLflow
    mlflow.set_tracking_uri(mlflow_params["tracking_uri"])
    mlflow.set_experiment(mlflow_params["experiment_name"])

    with mlflow.start_run():
        # Log params
        mlflow.log_params(train_params)

        # Train model
        model = RandomForestClassifier(
            n_estimators=train_params["n_estimators"],
            max_depth=train_params["max_depth"],
            random_state=train_params["random_state"]
        )
        model.fit(X_train, y_train)

        # Evaluate on train set
        train_preds = model.predict(X_train)
        train_acc = accuracy_score(y_train, train_preds)
        train_f1 = f1_score(y_train, train_preds)

        # Log metrics
        mlflow.log_metric("train_accuracy", train_acc)
        mlflow.log_metric("train_f1", train_f1)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Save model locally for DVC
        os.makedirs("models", exist_ok=True)
        with open("models/model.pkl", "wb") as f:
            pickle.dump(model, f)

        # Save run_id for evaluate step
        with open("models/run_id.txt", "w") as f:
            f.write(mlflow.active_run().info.run_id)

        print(f"✅ Model trained | Accuracy: {train_acc:.3f} | F1: {train_f1:.3f}")
        print(f"📝 MLflow Run ID: {mlflow.active_run().info.run_id}")

if __name__ == "__main__":
    train()