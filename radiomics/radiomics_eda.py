import pandas as pd


def eda_radiomics_features(csv_path: str):
    df = pd.read_csv(csv_path)
    print(df.describe())

if __name__ == "__main__":
    csv_path = r"Data/radiomics_features.csv"
    eda_radiomics_features(csv_path)