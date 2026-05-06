import pandas as pd
import yaml
import sys
import os

def load_config():
    with open("src/config.yaml", "r") as f:
        return yaml.safe_load(f)

def main():
    config = load_config()
    raw_path = config["data"]["raw"]
    output_path = config["prepare"]["output"]

    df = pd.read_csv(raw_path)

    df.drop_duplicates(inplace=True)

    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    df.drop(['Ticket', 'Cabin', 'Name'], axis=1, inplace=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Cleaned dataset saved to {output_path}")

if __name__ == "__main__":
    main()
