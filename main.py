from src.utils import load_config
from src.load_and_process_data import load_data_test, create_dummies, save_df
from src.model_training import train_estimator
from pathlib import Path
import pandas as pd
import mlflow
from mlflow.pyfunc import PythonModel

class EconmlModelWrapper(PythonModel):
    def __init__(self, estimator):
        self._estimator = estimator
    
    def get_estimator(self):
        return self._estimator
    
    def predict(self, context, X: pd.DataFrame) -> pd.Series:
        return self._estimator.const_marginal_effect(X)

def main():
    #load the yaml configuration file
    config = load_config()

    df = load_data_test(config)
    df_prc = create_dummies(df, ['weekday'])
    print(df_prc.columns)

    m1 = train_estimator(config, df_prc)
    #print(m1)

    features = config['model']['features']

    #estimate ATE
    ATE = m1.const_marginal_ate(df_prc[features])
    print(f'ATE: {ATE:.3f}')

    mlflow.set_tracking_uri('http://127.0.0.1:5000')
    mlflow.set_experiment('icecream-causal')

    with mlflow.start_run(run_name='causal') as run:

        model_info = mlflow.pyfunc.log_model(
            python_model=
        )

if __name__ == "__main__":
    main()
