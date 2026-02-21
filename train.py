import mlflow 
from load_data import * 
import pickle 
from sklearn.metrics import * 
import numpy as np 

df=load_data(data) 
df=round_data(df)
x,y=divide_data(df)
x_train,x_test,y_train,y_test = train_test_data(df)

mlflow.set_experiment('California House Price Predictions')

best_r2=-np.inf
best_run_id = None 
with open ('model.pkl','rb') as file:
    model=pickle.load(file)

    with mlflow.start_run():
        model.fit(x_train,y_train) 
        y_pred=model.predict(x_test) 

        r2=r2_score(y_test,y_pred) 
        # mlflow.log_param('n_estimators',n) 
        mlflow.log_metric('r2_score',r2)
        mlflow.sklearn.log_model(model,'house_model') 

        if r2>best_r2:
            best_r2=r2 
            best_run_id=mlflow.active_run().info.run_id 

print('Best run id',best_run_id)


    
