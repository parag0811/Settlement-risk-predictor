import numpy as np
import pandas as pd
from datetime import datetime, timedelta

np.random.seed(42)

def generate_raw_settlement_data(
    n_users=300,
    min_settlements=15,
    max_settlements=40
):
    rows = []
    base_date = datetime(2024, 1, 1)

    for user_id in range(1, n_users + 1):

        # Each user has a behavior tendency (fixed personality)
        user_delay_tendency = np.random.beta(2, 3)

        n_settlements = np.random.randint(min_settlements, max_settlements)

        for _ in range(n_settlements):

            amount = np.round(np.random.gamma(2.0, 250.0), 2)

            created_at = base_date + timedelta(
                days=np.random.randint(0, 365)
            )

            risk_score = (
                2.8 * user_delay_tendency +
                0.0018 * amount
            )

            prob_settle_fast = 1 / (1 + np.exp(risk_score - 3.5))

            settled_fast = np.random.binomial(1, prob_settle_fast)

            if settled_fast:
                delay_days = np.random.randint(0, 3)
            else:
                delay_days = np.random.randint(4, 12)

            settled_at = created_at + timedelta(days=delay_days)

            rows.append([
                user_id,
                amount,
                created_at,
                settled_at,
                1  # all generated settlements are completed
            ])

    df = pd.DataFrame(rows, columns=[
        "user_id",
        "amount",
        "created_at",
        "settled_at",
        "is_settled"
    ])

    return df


if __name__ == "__main__":
    df = generate_raw_settlement_data()
    df.to_csv("raw_settlement_data.csv", index=False)
    print("Raw dataset generated successfully!")
