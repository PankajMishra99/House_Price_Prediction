import pandas as pd 
import pydantic 
from pydantic import BaseModel
from typing import List,Literal 
from fastapi import FastAPI ,HTTPException
import pickle 

app=FastAPI()

with open('model.pkl','rb') as file:
    model=pickle.load(file)

class Userin(BaseModel):
    MedInc:float 
    HouseAge:float   
    AveRooms:float   
    AveBedrms: float     
    Population:float     
    AveOccup:float   
    

@app.get('/')  
def home():
    return {'message':'Welcome ! '}

@app.post('/predict')
def predict(data:Userin):
    input_df = pd.DataFrame([
        {
            'MedInc': data.MedInc,
            'HouseAge':data.HouseAge,
            'AveRooms':data.AveRooms,
            'AveBedrms': data.AveBedrms,
            'Population': data.Population,
            'AveOccup': data.AveOccup
           

        }
    ])


    prediction = model.predict(input_df)[0]
    return {'Prediction Price': prediction}





