# Settlement Risk Predictor

An ML-powered module that predicts the probability of a **delayed settlement** in group expense management apps (e.g. Splitwise-style tools), based on a user's historical settlement behavior. Exposed as a FastAPI service.

## How It Works

1. **Synthetic data generation** (`generate_data.py`) — creates a raw dataset of settlements per user, where each user has a fixed "delay tendency" plus amount-based risk, used to simulate `created_at` / `settled_at` timestamps.
2. **Feature engineering** (`notebooks/build_dataset.ipynb`) — converts raw settlements into per-transaction historical features (see below), using only *past* data at each point in time to avoid leakage.
3. **Model training** (`training/train.py`, `notebooks/final_model.ipynb`) — scales features with `StandardScaler` and fits a `LogisticRegression` classifier to predict `delayed_flag` (1 if a settlement took more than 3 days).
4. **Serving** (`src/app/`) — a FastAPI app loads the trained model/scaler and exposes a `/predict` endpoint that returns a delay probability and a risk bucket.

## Features Used by the Model

| Feature | Description |
|---|---|
| `amount` | Settlement amount |
| `past_total_settlements` | Number of settlements the user has made before this one |
| `past_delay_rate` | Fraction of the user's past settlements that were delayed |
| `avg_settlement_time` | User's average delay (in days) across past settlements |
| `past_avg_amount` | User's average settlement amount so far |
| `amount_vs_user_avg` | Ratio of current amount to the user's past average amount |
| `avg_delay_last_5` | Average delay over the user's last 5 settlements |

## Project Structure

```
Settlement-risk-predictor/
├── generate_data.py          # Synthetic raw settlement data generator
├── notebooks/
│   ├── build_dataset.ipynb          # Raw data -> engineered features
│   ├── training_Logistic_regression.ipynb
│   ├── training_random_forest.ipynb
│   ├── training_model_kfold.ipynb
│   └── final_model.ipynb            # Final training run used for the deployed model
├── training/
│   └── train.py               # Standalone training script (scaler + logistic regression)
├── models/
│   ├── quicksplit_logistic_model.pkl
│   └── quicksplit_scaler.pkl
├── src/app/
│   ├── main.py                 # FastAPI app + /predict route
│   ├── predictor.py             # Loads model/scaler, runs inference, buckets risk
│   └── schema.py
├── requirements.txt
└── Procfile                     # `uvicorn src.app.main:app`
```

## API

### `GET /`
Health check.
```json
{ "message": "Settlement Risk API is running..." }
```

### `POST /predict`
**Request body:**
```json
{
  "amount": 500,
  "past_total_settlements": 10,
  "past_delay_rate": 0.3,
  "avg_settlement_time": 2.5,
  "past_avg_amount": 450,
  "amount_vs_user_avg": 1.1,
  "avg_delay_last_5": 2.0
}
```

**Response:**
```json
{
  "delay_probability": 0.42,
  "risk_level": "Medium"
}
```

Risk is bucketed from the predicted probability:
- `> 0.7` → **High**
- `> 0.4` → **Medium**
- otherwise → **Low**

## Setup

```bash
git clone https://github.com/parag0811/Settlement-risk-predictor.git
cd Settlement-risk-predictor
pip install -r requirements.txt
uvicorn src.app.main:app --reload
```

The API will be available at `https://expense-anomaly-ml-service.onrender.com`, with interactive docs at `/docs`.

## Regenerating the Model

```bash
python generate_data.py                       # writes raw_settlement_data.csv
# run notebooks/build_dataset.ipynb to produce data/processed/final_dataset.pkl
python training/train.py                      # fits scaler + logistic regression, saves .pkl files
```

## Notes

- Training data is currently **synthetic** (generated via `generate_data.py`), used to prototype the pipeline end-to-end before wiring in real transaction history.
- `notebooks/` also contains experiments with Random Forest and k-fold cross-validation; the model shipped in `models/` (and used by the API) is the **Logistic Regression** version.
