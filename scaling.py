import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib   

df = pd.read_csv("datasets/f1_encoded_not_scaled.csv")


scale_cols = [
    "track_length",
    "driver_perf",
    "team_perf",
    "fp1_long_run",
    "fp2_long_run",
    "fp3_long_run",
    "race_pace"
]


scaler = StandardScaler()
df[scale_cols] = scaler.fit_transform(df[scale_cols])

df.to_csv("f1_encoded_scaled.csv", index=False)

joblib.dump(scaler, "scaler.pkl")
