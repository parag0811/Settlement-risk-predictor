import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_pickle("../../data/processed/final_dataset.pkl")
df

X = df.drop(['delayed_flag'], axis='columns')
y = df['delayed_flag']

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
scaler.fit(X)
X_scaled = scaler.transform(X)


import joblib
joblib.dump(scaler, "quicksplit_scaler.pkl")

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1000)
model.fit(X_scaled, y)
joblib.dump(model, "quicksplit_logistic_model.pkl")
