import pandas as pd
import yaml
import os

def extract_title(name):
    if pd.isna(name):
        return "Unknown"
    return name.split(",")[1].split(".")[0].strip()

def main():
    config = yaml.safe_load(open("src/config.yaml"))
    input_path = config["data"]["raw"]
    output_path = config["featurize"]["output"]

    df = pd.read_csv(input_path)

    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)
    df['Title'] = df['Name'].apply(extract_title)

    df.drop(['Ticket', 'Cabin', 'Name', 'SibSp', 'Parch'], axis=1, inplace=True)

    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if 'Survived' in categorical_cols:
        categorical_cols.remove('Survived')

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Feature dataset with encoding saved to {output_path}")
    print("Columns:", df.columns.tolist())

if __name__ == "__main__":
    main()
