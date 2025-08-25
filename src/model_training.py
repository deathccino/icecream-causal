from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from pathlib import Path
import mlflow
from load_and_process_data import load_data
from utils import load_config

class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, estimator: CausalForestDML):
        self._estimator = estimator
    
    def get_estimator(self):
        return self._estimator
    
    def predict(self, model_input: pd.DataFrame) -> np.ndarray:
        return self._estimator.const_marginal_effect(model_input)

def train_estimator(config: dict, df: pd.DataFrame):

    model_details = config['model']

    model = CausalForestDML(
        model_t=GradientBoostingRegressor(),
        model_y=GradientBoostingRegressor(),
        cv=3,
        random_state=1
    )

    model.fit(
        Y=df[model_details['outcome']],
        T=df[model_details['treatment']],
        X=df[model_details['features']]
    )

    return model

def log_mlflow_model(model, input_example):
    model_info = mlflow.pyfunc.log_model(
        name='Causal-Forest',
        python_model=ModelWrapper(model),
        input_example=input_example
    )
    return model_info

def main():
    config = load_config()
    filepath = config['data']['path'] + r'\\processed'

    df_train = load_data(filepath , 'ice_cream_train.csv')
    model = train_estimator(config, df_train)
    features = config['model']['features']

    mlflow.set_tracking_uri(config['mlflow']['uri'])
    mlflow.set_experiment(experiment_name=config['mlflow']['experiment_name'])

    with mlflow.start_run() as run:

        #log the econml estimator
        model_info = log_mlflow_model(model, input_example=df_train.loc[[0], features])

if __name__ == '__main__':
    main()