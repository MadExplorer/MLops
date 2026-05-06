import pandas as pd
import yaml
import joblib
import json
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def main():
    config = yaml.safe_load(open("src/config.yaml"))
    test_path = config["evaluate"]["testset"]
    model_path = config["evaluate"]["model_path"]
    metrics_file = config["evaluate"]["metrics_file"]

    test = pd.read_csv(test_path)
    y_true = test["Survived"]
    X_test = test.drop("Survived", axis=1)

    model = joblib.load(model_path)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    import os
    os.makedirs(os.path.dirname(metrics_file), exist_ok=True)
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    print("Evaluation metrics:", json.dumps(metrics, indent=2))

if __name__ == "__main__":
    main()
