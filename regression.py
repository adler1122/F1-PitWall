import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import GradientBoostingRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from scipy.stats import spearmanr

df = pd.read_csv("datasets/f1_encoded_scaled.csv")

if "race_pace" in df.columns:
    df = df.drop(columns=["race_pace"])


X = df.drop(columns=["race_id", "relative_finish"])
y = df["relative_finish"] 


race_ids = df["race_id"].unique()
train_races = race_ids[:-19]
test_races = race_ids[-19:]

X_train = X[df["race_id"].isin(train_races)].reset_index(drop=True)
y_train = y[df["race_id"].isin(train_races)].reset_index(drop=True)

X_test = X[df["race_id"].isin(test_races)].reset_index(drop=True)
y_test = y[df["race_id"].isin(test_races)].reset_index(drop=True)

df_test = df[df["race_id"].isin(test_races)].reset_index(drop=True)

models = {
    "Linear Regression": LinearRegression(),
    "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, learning_rate=0.05, max_depth=5),
    "LightGBM": lgb.LGBMRegressor(objective="regression", n_estimators=200, learning_rate=0.05, num_leaves=31),
}

predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    predictions[name] = y_pred

    mse = mean_squared_error(y_test, y_pred)
    corr, _ = spearmanr(y_test, y_pred)
    print(f"{name} MSE: {mse:.4f}, Spearman: {corr:.4f}")

print("\nTraining XGBoost Regressor...")
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)

params = {
    "objective": "reg:squarederror",
    "eta": 0.05,
    "max_depth": 6,
    "eval_metric": "rmse"
}

xgb_reg = xgb.train(params, dtrain, num_boost_round=200, evals=[(dtest, "test")])
y_pred_xgb = xgb_reg.predict(dtest)
predictions["XGBoost"] = y_pred_xgb

mse_xgb = mean_squared_error(y_test, y_pred_xgb)
corr_xgb, _ = spearmanr(y_test, y_pred_xgb)
print(f"XGBoost MSE: {mse_xgb:.4f}, Spearman: {corr_xgb:.4f}")



for race in test_races:
    race_results = df_test[df_test["race_id"] == race].copy()
    true_order = race_results.sort_values("relative_finish")["relative_finish"].values

    print(f"\nRace {race}:")
    for name, y_pred in predictions.items():
        race_results[f"pred_{name}"] = y_pred[race_results.index]
        pred_order = race_results.sort_values(f"pred_{name}")["relative_finish"].values
        accuracy = (true_order == pred_order).sum()
        print(f"{name} correct = {accuracy}")