

from fastapi import FastAPI
from pydantic import BaseModel
import torch
import uvicorn
from model import Model
from engine import predict

app = FastAPI()
model_path = 'model.w'
model = Model()
model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))

class Input(BaseModel):
    symbol: str
    num_symbol: int

class Output(BaseModel):
    symbol_price: list
    topN: dict

@app.get("/")
def read_root():
    return {"Hello": "World"}


@app.post("/predict/", response_model = Output)
def response_predict(input: Input):
    symbol_price, topN = predict(model, input.symbol, input.num_years)
    
    return {
        'symbol_price': symbol_price,
        'topN': topN
    }
    
if __name__ == "__main__":
    config = uvicorn.Config("api:app", port=5000)
    server = uvicorn.Server(config)
    server.run()