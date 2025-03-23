from fastapi import FastAPI, Request, Body
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import numpy as np
import uvicorn
import pickle

app = FastAPI()

with open("xgboost.pkl", "rb") as model_file:
    model = pickle.load(model_file)

# from pydantic import BaseModel

# class InsuranceData(BaseModel):
#     age: int
#     sex: int  
#     bmi: float
#     children: int
#     smoker: int  
#     region: int 

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def serve_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(
    age: int = Body(...),
    sex: int = Body(...),
    bmi: float = Body(...),
    children: int = Body(...),
    smoker: int = Body(...),
    region: int = Body(...)
):
    input_data = np.array([[age, sex, bmi, children, smoker, region]])
    prediction = model.predict(input_data)[0]

    return {"predicted_insurance_cost": round(float(prediction), 2)}


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)
