from fastapi import FastAPI
from pydantic import BaseModel
from .predictor import predict_settlement_risk

app = FastAPI(
    title="QuickSplit Settlement Risk API",
    version="1.0"
)

class SettlementFeatures(BaseModel):
    amount: float
    past_total_settlements: int
    past_delay_rate: float
    avg_settlement_time: float
    past_avg_amount: float
    amount_vs_user_avg: float
    avg_delay_last_5: float

@app.get("/")
def home():
    return {"message": "Settlement Risk API is running..."}

@app.post("/predict")
def predict(data: SettlementFeatures):

    result = predict_settlement_risk(
        amount=data.amount,
        past_total_settlements=data.past_total_settlements,
        past_delay_rate=data.past_delay_rate,
        avg_settlement_time=data.avg_settlement_time,
        past_avg_amount=data.past_avg_amount,
        amount_vs_user_avg=data.amount_vs_user_avg,
        avg_delay_last_5=data.avg_delay_last_5
    )

    return result