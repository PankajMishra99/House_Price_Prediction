import mlflow 

run_id='5675a4cf464247dfb6ac49d55c3e51a7'
model_uri=f"runs:/{run_id}/house_model" 

mlflow.register_model(
    model_uri=model_uri,
    name='House_Price_Model'
)