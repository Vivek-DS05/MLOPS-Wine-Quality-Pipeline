import pandas as pd
from sklearn.model_selection import train_test_split
import yaml
import os

def prepare():
    # Load params
    with open("params.yaml") as f:
        params = yaml.safe_load(f)["prepare"]

    # Download wine dataset from URL
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
    df = pd.read_csv(url, sep=";")

    # Simple feature engineering
    df["quality_binary"] = (df["quality"] >= 6).astype(int)
    df.drop("quality", axis=1, inplace=True)

    # Split data
    train, test = train_test_split(
        df,
        test_size=params["test_size"],
        random_state=params["random_state"]
    )

    # Save
    os.makedirs("data", exist_ok=True)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    print(f"✅ Data prepared: {len(train)} train, {len(test)} test samples")

if __name__ == "__main__":
    prepare()