import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

MODEL_PATH = os.path.join(BASE_DIR, "models", "quicksplit_logistic_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "quicksplit_scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

FEATURE_ORDER = [
    "amount",
    "past_total_settlements",
    "past_delay_rate",
    "avg_settlement_time",
    "past_avg_amount",
    "amount_vs_user_avg",
    "avg_delay_last_5"
]


def predict_settlement_risk(
    amount,
    past_total_settlements,
    past_delay_rate,
    avg_settlement_time,
    past_avg_amount,
    amount_vs_user_avg,
    avg_delay_last_5
):
    features = np.array([[
        amount,
        past_total_settlements,
        past_delay_rate,
        avg_settlement_time,
        past_avg_amount,
        amount_vs_user_avg,
        avg_delay_last_5
    ]])

    scaled_features = scaler.transform(features)

    probability = model.predict_proba(scaled_features)[0][1]

    if probability > 0.7:
        risk_level = "High"
    elif probability > 0.4:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "delay_probability": float(probability),
        "risk_level": risk_level
    }


if __name__ == "__main__":
    result = predict_settlement_risk(   
        amount=500,
        past_total_settlements=10,
        past_delay_rate=0.3,
        avg_settlement_time=2.5,
        past_avg_amount=450,
        amount_vs_user_avg=1.1,
        avg_delay_last_5=2.0
    )

    print(result)