import mlflow
import pandas as pd
from mlflow.pyfunc import PythonModel
from mlflow.models import set_model

class EconmlModelWrapper(PythonModel):
    def __init__(self, estimator):
        self._estimator = estimator
    
    def get_estimator(self):
        return self._estimator
    
    def predict(self, context, X: pd.DataFrame) -> pd.Series:
        return self._estimator.const_marginal_effect(X)

set_model(EconmlModelWrapper)