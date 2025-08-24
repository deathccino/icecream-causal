#%%
import numpy as np
import pandas as pd
import re
from pathlib import Path
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
# %%
"""
TO-DO functions:
    load data;
    feature engineering (dummy);
    save processed data
"""
# %%
def load_data_train(config: dict) -> pd.DataFrame:
    file_path = config['data']['path']
    df = pd.read_csv(file_path)
    return df

def load_data_test(config: dict):
    file_path = config['data']['path_rnd']
    df = pd.read_csv(file_path)
    return df
# %%
def create_dummies(df: pd.DataFrame, features):
    preprocessor = ColumnTransformer(
        transformers=[
            ('dummies', OneHotEncoder(sparse_output=False, dtype=int), features)
        ],
        remainder='passthrough'
    ).fit(df[features])

    cols = [re.sub('.*__', '', col) for col in preprocessor.get_feature_names_out()]
    df_prc = pd.DataFrame(
        data=preprocessor.transform(df[features]),
        columns=cols
    )

    return df_prc
# %%
def save_df(config: dict, df: pd.DataFrame, filename: str):
    filepath = Path(config['data']['processed']) / filename
    df.to_csv(path_or_buf=filepath)