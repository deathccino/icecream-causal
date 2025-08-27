from econml.dml import CausalForestDML
from sklearn.ensemble import GradientBoostingRegressor
import numpy as np
import pandas as pd
from pathlib import Path
import mlflow
import matplotlib.pyplot as plt
from load_and_process_data import load_data
from utils import load_config, sensitivity, sensitivity_cum, calc_AUUC

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

def plot_uplift_curve(model: CausalForestDML, df: pd.DataFrame, features: list[str], outcome: str, treatment: str):

    y_pred = model.const_marginal_effect(df[features])
    df_c = df.assign(pred=y_pred, rnd=np.random.uniform(size=df.shape[0]))

    #calculate sensitivity
    X, y = sensitivity_cum(df_c, 'pred', outcome, treatment)
    _, y_rnd = sensitivity_cum(df_c, 'rnd', outcome, treatment)
    full_sensitivity = sensitivity(df_c, outcome, treatment)

    fig, axs = plt.subplots(figsize=(8, 6))
    axs.plot(X, y_rnd, label='Random')
    axs.plot(X, y, label='Model')

    axs.set_xlabel('samples')
    axs.set_ylabel('gain')
    axs.set_title(f'Dataset sensitivity: {full_sensitivity:.3f}')

    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.legend()
    fig.savefig('artifacts/chart.png')

    return X, y, y_rnd

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

        #log the uplift curve
        X, y, y_rnd = plot_uplift_curve(model, df_train, features, config['model']['outcome'], config['model']['treatment'])
        auuc_model = calc_AUUC(X, y)
        print(f'AUUC Model: {auuc_model:.3f}')

        mlflow.log_artifact('./artifacts/chart.png', artifact_path='charts')
        mlflow.log_metric("AUUC", auuc_model)


        mlflow.log_params(
            {
                'treatment': config['model']['treatment'],
                'outcome': config['model']['outcome'],
                'features': config['model']['features'],
                'estimator_type': 'CausalForest'
             }
        )



if __name__ == '__main__':
    main()