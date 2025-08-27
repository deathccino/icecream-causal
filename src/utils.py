import yaml
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def load_config():
    config_path = Path(__file__).parent.parent / 'config.yaml'
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    
    return config

def sensitivity(data: pd.DataFrame, y, t):
    # line coeficient for the one variable linear regression 
    x1 = data[t].cov(data[y])
    x2 = data[t].var()
    return x1/x2

def sensitivity_cum(df: pd.DataFrame, column: str, outcome: str, treatment: str, normalize: bool = True):
    ordering = np.argsort(-df[column])
    df_ordered = df.loc[ordering, :].copy()
    df_size = len(df_ordered)
    batch_list = np.array(np.arange(start=30, stop=df_size+1))
    if normalize:
        sensi_list = np.array([sensitivity(df_ordered.head(batch), outcome, treatment)*(batch/df_size) for batch in batch_list])
        return batch_list/df_size, sensi_list
    else:
        sensi_list = np.array([sensitivity(df_ordered.head(batch), outcome, treatment) for batch in batch_list])
        return batch_list, sensi_list

def calc_AUUC(X, y):
    return np.trapezoid(x=X, y=y-y[-1])