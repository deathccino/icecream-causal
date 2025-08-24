from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from pathlib import Path

def train_estimator(config: dict, filename: str):
    filepath = Path(config['data']['processed']) / filename
    model_details = config['model']
    df = pd.read_csv(filepath)

    m1 = CausalForestDML(
        model_t=GradientBoostingRegressor(),
        model_y=GradientBoostingRegressor(),
        cv=3,
        random_state=1
    )

    m1.fit(
        Y=df[model_details['outcome']],
        T=df[model_details['treatment']],
        X=df[model_details['features']]
    )

    return m1