import pandas as pd
import yaml
from sklearn.model_selection import train_test_split

def main():
    config = yaml.safe_load(open("src/config.yaml"))
    input_path = config["featurize"]["output"]
    test_size = config["data_split"]["test_size"]
    train_path = config["data_split"]["trainset"]
    test_path = config["data_split"]["testset"]

    df = pd.read_csv(input_path)
    train, test = train_test_split(df, test_size=test_size, random_state=42)

    train.to_csv(train_path, index=False)
    test.to_csv(test_path, index=False)
    print(f"Train: {train.shape[0]} rows, Test: {test.shape[0]} rows")

if __name__ == "__main__":
    main()
