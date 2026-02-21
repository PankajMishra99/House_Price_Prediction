import pandas as pd    
from sklearn.model_selection import train_test_split 
from sklearn.ensemble import RandomForestRegressor 
# import mlflow 
from sklearn.metrics import * 
from sklearn.datasets import fetch_california_housing  
from sklearn.compose import ColumnTransformer 
from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler 
import pickle 

data=fetch_california_housing()

def load_data(data):
    df_input=pd.DataFrame(
        data.data,
        columns=data.feature_names

    )
    df_input.drop(['Latitude','Longitude'],inplace=True,axis=1)
    df_output =pd.DataFrame(data.target,columns=data.target_names)
    df=pd.concat([df_input,df_output],axis=1)
    return df  

def round_data(df:pd.DataFrame): 
    for col in df.columns:
        df[col]=df[col].round(2)
    return df  

def divide_data(df:pd.DataFrame):
    df=round_data(df)
    x=df.drop('MedHouseVal',axis=1)
    y=df['MedHouseVal']
    return x,y


def train_test_data(df:pd.DataFrame):
    x,y=divide_data(df)
    x_train,x_test,y_train,y_test= train_test_split(x,y,test_size=0.2,random_state=42)
    return x_train,x_test,y_train,y_test 

def preprocess_data(df:pd.DataFrame):
    x,y=divide_data(df)
    process = ColumnTransformer(
        transformers=[
            ('num',StandardScaler(),x.columns)
        ]
    )
    return process 

def pipline_data(df:pd.DataFrame):
    process=preprocess_data(df) 
    x_train,x_test,y_train,y_test = train_test_data(df)
    pipline_step=Pipeline([
        ('preporcess',process),
        ('model',RandomForestRegressor())
    ])
    pipline_step.fit(x_train,y_train)
    return pipline_step 

def predict_data(df:pd.DataFrame):
    x_train,x_test,y_train,y_test = train_test_data(df)
    pipline = pipline_data(df)
    y_pred= pipline.predict(x_test)
    rmse = root_mean_squared_error(y_test,y_pred) 
    r2_score=r2_score(y_test,y_pred) 

def save_model(df):
    model=pipline_data(df) 
    with open('model.pkl','wb') as file:
        pickle.dump(model,file) 

    

def main():
    df=load_data(data)
    df=round_data(df)
    x,y=divide_data(df)
    x_train,x_test,y_train,y_test=train_test_data(df)
    process=preprocess_data(df)
    pipline=pipline_data(df)
    model=save_model(df) 
    return model 

# print(main())