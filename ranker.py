import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import ndcg_score
from scipy.stats import spearmanr


df = pd.read_csv("datasets/f1_encoded_not_scaled.csv")

df = df.drop(columns=["race_pace"])
df["rank_label"] = df.groupby("race_id")["relative_finish"].rank(method="first").astype(int)

X = df.drop(columns=["race_id", "relative_finish", "rank_label"])
y = df["rank_label"]


race_ids = df["race_id"].unique()
train_races = race_ids[:-19]
test_races = race_ids[-19:]

X_train = X[df["race_id"].isin(train_races)]
y_train = y[df["race_id"].isin(train_races)]
groups_train = df[df["race_id"].isin(train_races)].groupby("race_id").size().to_list()

X_test = X[df["race_id"].isin(test_races)]
y_test = y[df["race_id"].isin(test_races)]
groups_test = df[df["race_id"].isin(test_races)].groupby("race_id").size().to_list()


print("training LightGBM Ranker")
lgb_ranker = lgb.LGBMRanker(
    objective="lambdarank",
    metric="ndcg",
    boosting_type="gbdt",
    n_estimators=200,
    learning_rate=0.05,
    num_leaves=31,
    verbose=1
)

lgb_ranker.fit(
    X_train, y_train,
    group=groups_train,
    eval_set=[(X_test, y_test)],
    eval_group=[groups_test],
    eval_at=[5, 10]
)

y_pred_lgb = lgb_ranker.predict(X_test)

ndcg_lgb = ndcg_score([y_test.values], [y_pred_lgb])
corr_lgb, _ = spearmanr(y_test, y_pred_lgb)
print(f"LightGBM NDCG: {ndcg_lgb:.4f}, Spearman: {corr_lgb:.4f}")

print("training XGBoost Ranker")
dtrain = xgb.DMatrix(X_train, label=y_train)
dtest = xgb.DMatrix(X_test, label=y_test)


dtrain.set_group(groups_train)
dtest.set_group(groups_test)

params = {
    "objective": "rank:pairwise",
    "eval_metric": "ndcg",
    "eta": 0.05,
    "max_depth": 6
}

xgb_ranker = xgb.train(
    params,
    dtrain,
    num_boost_round=200,
    evals=[(dtest, "test")]
)

y_pred_xgb = xgb_ranker.predict(dtest)

ndcg_xgb = ndcg_score([y_test.values], [y_pred_xgb])
corr_xgb, _ = spearmanr(y_test, y_pred_xgb)
print(ndcg_xgb, corr_xgb)


df_test = df[df["race_id"].isin(test_races)].copy()
df_test["pred_finish_lgb"] = y_pred_lgb
df_test["pred_finish_xgb"] = y_pred_xgb

for race in test_races:
    race_results = df_test[df_test["race_id"] == race].copy()

    
    true_order = race_results.sort_values("rank_label")["rank_label"].values

    
    lgb_order = race_results.sort_values("pred_finish_lgb")["rank_label"].values
    lgb_accuracy = (true_order == lgb_order).sum()

    
    xgb_order = race_results.sort_values("pred_finish_xgb")["rank_label"].values
    xgb_accuracy = (true_order == xgb_order).sum()

    print(f"race {race}: LightGBM correct = {lgb_accuracy}, XGBoost correct = {xgb_accuracy}")