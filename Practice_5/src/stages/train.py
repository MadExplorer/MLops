import pandas as pd
import yaml
import joblib
from sklearn.ensemble import RandomForestClassifier

def main():
    config = yaml.safe_load(open("src/config.yaml"))
    train_path = config["data_split"]["trainset"]
    model_path = config["train"]["model_path"]

    df = pd.read_csv(train_path)
    y = df["Survived"]
    X = df.drop("Survived", axis=1)

    model = RandomForestClassifier(
        n_estimators=config["train"]["n_estimators"],
        max_depth=config["train"]["max_depth"],
        random_state=42
    )
    model.fit(X, y)

    import os
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

if __name__ == "__main__":
    main()
