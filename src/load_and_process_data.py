import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from utils import load_config

def load_data(filepath: str, filename: str) -> pd.DataFrame:
    return pd.read_csv(Path(filepath) / filename)

def save_df(df: pd.DataFrame, filepath: str, filename: str):
    df.to_csv(Path(filepath) / filename, index=False)

def fit_preprocessor(df: pd.DataFrame, features: list) -> ColumnTransformer:
    preprocessor = ColumnTransformer(
        transformers=[
            ('dummies', OneHotEncoder(sparse_output=False, dtype=int), features)
        ],
        remainder='passthrough'
    ).fit(df)

    return preprocessor

def create_dummies(df: pd.DataFrame, preprocessor: ColumnTransformer) -> pd.DataFrame:
    #perhaps I should serialize the fitted preprocessor?
    prc = preprocessor
    cols = [re.sub('.*__', '', col) for col in prc.get_feature_names_out()]
    df_prc = pd.DataFrame(
        data=prc.transform(df),
        columns=cols
    )

    return df_prc

def main():
    #load the yaml configuration file
    config = load_config()

    filepath = config['data']['path']

    #load and process train data
    df_train = load_data(filepath + r'\\raw', 'ice_cream_train.csv')
    prc = fit_preprocessor(df_train, ['weekday'])
    save_df(create_dummies(df_train, prc), filepath + r'\\processed', filename='ice_cream_train.csv')
    print("Train data processed and saved sucessfully")

    #load and process test data
    df_test = load_data(filepath + r'\\raw', 'ice_cream_test.csv')
    save_df(create_dummies(df_test, prc), filepath + r'\\processed', filename='ice_cream_test.csv')
    print("Test data processed and saved sucessfully")

if __name__ == '__main__':
    main()