import mlflow 
from mlflow.tracking import MlflowClient 

client=MlflowClient()
client.transition_model_version_stage(
    name='House_Price_Model',
    version=1,
    stage='Production'
)