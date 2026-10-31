"""
Predictive Modeling and Optimization in Logistics Systems (Week 4)

Input : cleaned_logistics_data.csv  (created by data_cleaning.py in Week 2)
Output: results printed to the screen and chart images (model*.png)

PART A  Predict actual shipping days for an order (regression)
        - compares a baseline, linear regression, decision tree, random forest
          and gradient boosting
        - cross-validation, hyperparameter tuning, RMSE / MAE / R-squared
        - uses the model results to optimise the delivery time promised to customers
PART B  Forecast daily order volume (time series regression)
        - uses the forecast to plan vehicle capacity

Install:  pip install pandas numpy matplotlib seaborn scikit-learn
Run:      python predictive_modeling.py
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import (train_test_split, cross_validate, GridSearchCV,
                                     KFold, TimeSeriesSplit)
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.inspection import permutation_importance

CLEAN_FILE = sys.argv[1] if len(sys.argv) > 1 else "cleaned_logistics_data.csv"
SEED = 42
sns.set_theme(style="whitegrid")

REAL = "Days for shipping (real)"
SCHED = "Days for shipment (scheduled)"
MODE = "Shipping Mode"
DATE = "order date (DateOrders)"

# assumption used only in the capacity planning example in Part B (change if you like)
DELIVERIES_PER_VEHICLE_DAY = 40

summary = {}


def title(text):
    print("\n" + "=" * 64)
    print(text)
    print("=" * 64)


def metrics(y_true, y_pred, mape=False):
    out = {"MAE": mean_absolute_error(y_true, y_pred),
           "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
           "R2": r2_score(y_true, y_pred)}
    if mape:
        out["MAPE %"] = np.mean(np.abs(y_true - y_pred) / np.maximum(y_true, 1)) * 100
    return out


def save(name):
    plt.tight_layout()
    plt.savefig(name, dpi=150)
    plt.close()
    print("   saved", name)


# ================================================================ load
if not os.path.exists(CLEAN_FILE):
    sys.exit(f"Could not find '{CLEAN_FILE}'. Run data_cleaning.py first.")
df = pd.read_csv(CLEAN_FILE, low_memory=False)
df[DATE] = pd.to_datetime(df[DATE], errors="coerce")
print("Loaded", CLEAN_FILE, df.shape)


# ================================================================ PART A
title("PART A: PREDICTING ACTUAL SHIPPING DAYS")

# canceled orders have no real delivery, so they are left out
if "Delivery Status" in df.columns:
    n0 = len(df)
    d = df[df["Delivery Status"] != "Shipping canceled"].copy()
    print(f"Removed {n0 - len(d)} canceled order items; {len(d)} remain")
else:
    d = df.copy()

d["order_month"] = d[DATE].dt.month
d["order_weekday"] = d[DATE].dt.dayofweek

# Only information known when the order is placed is used as input.
# Delay, late flag and delivery status are NOT used, since they already contain the answer.
cat_feats = [c for c in [MODE, "Market", "Order Region", "Category Name", "Customer Segment"] if c in d.columns]
num_feats = [c for c in [SCHED, "Order Item Quantity", "Sales", "Order Item Discount Rate",
                         "order_month", "order_weekday"] if c in d.columns]
features = cat_feats + num_feats
print("Target:", REAL)
print("Categorical inputs:", cat_feats)
print("Numeric inputs    :", num_feats)

X, y = d[features], d[REAL]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
print(f"Training rows: {len(X_train)} | Test rows: {len(X_test)}")


def preprocessor(scale=False):
    num = StandardScaler() if scale else "passthrough"
    return ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_feats),
        ("num", num, num_feats)])


def make(model, scale=False):
    return Pipeline([("prep", preprocessor(scale)), ("model", model)])


# ---------------------------------------------------- hyperparameter tuning
title("A1. HYPERPARAMETER TUNING (grid search with cross-validation)")
n_tune = min(30000, len(X_train))
Xs, ys = X_train.sample(n_tune, random_state=SEED), y_train.loc[X_train.sample(n_tune, random_state=SEED).index]

tree_search = GridSearchCV(
    make(DecisionTreeRegressor(random_state=SEED)),
    {"model__max_depth": [4, 6, 8, 12], "model__min_samples_leaf": [20, 50, 100]},
    cv=3, scoring="neg_root_mean_squared_error", n_jobs=-1)
tree_search.fit(Xs, ys)
print("Decision tree best settings:", tree_search.best_params_,
      "| CV RMSE:", round(-tree_search.best_score_, 3))

rf_search = GridSearchCV(
    make(RandomForestRegressor(n_estimators=60, random_state=SEED, n_jobs=-1)),
    {"model__max_depth": [8, 12], "model__min_samples_leaf": [10, 30]},
    cv=3, scoring="neg_root_mean_squared_error")
rf_search.fit(Xs.iloc[:20000], ys.iloc[:20000])
print("Random forest best settings:", rf_search.best_params_,
      "| CV RMSE:", round(-rf_search.best_score_, 3))
summary["tree_params"] = tree_search.best_params_
summary["rf_params"] = rf_search.best_params_

tree_p = {k.replace("model__", ""): v for k, v in tree_search.best_params_.items()}
rf_p = {k.replace("model__", ""): v for k, v in rf_search.best_params_.items()}

models = {
    "Baseline (mean)": make(DummyRegressor(strategy="mean")),
    "Linear regression": make(LinearRegression(), scale=True),
    "Decision tree (tuned)": make(DecisionTreeRegressor(random_state=SEED, **tree_p)),
    "Random forest (tuned)": make(RandomForestRegressor(n_estimators=100, random_state=SEED, n_jobs=-1, **rf_p)),
    "Gradient boosting": make(HistGradientBoostingRegressor(max_iter=150, learning_rate=0.1, random_state=SEED)),
}

# ---------------------------------------------------- cross-validation
title("A2. 5-FOLD CROSS-VALIDATION (on a sample of the training data)")
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
cv_rows = {}
for name, pipe in models.items():
    res = cross_validate(pipe, Xs, ys, cv=kf,
                         scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"})
    cv_rows[name] = {"CV RMSE": -res["test_rmse"].mean(), "RMSE std": res["test_rmse"].std(),
                     "CV R2": res["test_r2"].mean()}
cv_table = pd.DataFrame(cv_rows).T.round(3)
print(cv_table.to_string())

# ---------------------------------------------------- final fit and test scores
title("A3. TEST SET RESULTS (data the models have never seen)")
rows, preds = {}, {}
for name, pipe in models.items():
    pipe.fit(X_train, y_train)
    p = pipe.predict(X_test)
    preds[name] = p
    rows[name] = metrics(y_test, p)
test_table = pd.DataFrame(rows).T.round(3)
print(test_table.to_string())
best_name = test_table["RMSE"].idxmin()
best_pipe = models[best_name]
print("\nBest model on the test set:", best_name)
summary["best_model"] = best_name
summary["test_table"] = test_table

# ---------------------------------------------------- feature importance
title("A4. WHICH INPUTS MATTER MOST (permutation importance)")
n_imp = min(10000, len(X_test))
Xi = X_test.sample(n_imp, random_state=SEED)
yi = y_test.loc[Xi.index]
imp = permutation_importance(best_pipe, Xi, yi, n_repeats=3, random_state=SEED,
                             scoring="neg_root_mean_squared_error", n_jobs=-1)
imp_s = pd.Series(imp.importances_mean, index=features).sort_values(ascending=False)
print("Increase in RMSE when the input is shuffled (bigger = more important):")
print(imp_s.round(3).to_string())
summary["top_feature"] = imp_s.index[0]

# ---------------------------------------------------- charts for part A
title("A5. CHARTS")
plt.figure(figsize=(7, 4.5))
cmp = test_table["RMSE"].sort_values()
ax = sns.barplot(x=cmp.values, y=cmp.index, color="#2a6f97")
for i, v in enumerate(cmp.values):
    ax.text(v + 0.01, i, f"{v:.3f}", va="center")
plt.xlabel("Test RMSE (days), lower is better"); plt.ylabel("")
plt.title("Model comparison on the test set")
save("model1_model_comparison.png")

plt.figure(figsize=(7, 4.5))
s = imp_s[::-1]
plt.barh(s.index, s.values, color="#e07a5f")
plt.xlabel("Increase in RMSE when shuffled")
plt.title("Importance of each input (" + best_name + ")")
save("model2_feature_importance.png")

plt.figure(figsize=(6, 5.5))
rng = np.random.default_rng(SEED)
idx = rng.choice(len(y_test), size=min(2500, len(y_test)), replace=False)
bp = preds[best_name]
plt.scatter(y_test.values[idx] + rng.normal(0, 0.08, len(idx)), bp[idx], alpha=0.3, s=10, color="#1d3557")
plt.plot([0, 6], [0, 6], "r--", label="Perfect prediction")
plt.xlabel("Actual shipping days"); plt.ylabel("Predicted shipping days")
plt.title("Actual vs predicted shipping days (" + best_name + ")"); plt.legend()
save("model3_actual_vs_predicted.png")

plt.figure(figsize=(7, 4.5))
resid = y_test.values - bp
sns.histplot(resid, bins=30, color="#52796f")
plt.axvline(0, color="red", linestyle="--")
plt.xlabel("Error (actual minus predicted days)"); plt.ylabel("Number of order items")
plt.title("Prediction errors (" + best_name + ")")
save("model4_residuals.png")

# ---------------------------------------------------- optimisation: promised time
title("A6. OPTIMISATION: CHOOSING THE DELIVERY TIME TO PROMISE")
print("Current promise vs a promise set from the actual delivery times of each mode.")
print("(An order is late if the real days are more than the promised days.)\n")
opt_rows = []
for m, sub in d.groupby(MODE):
    cur = sub[SCHED].round().mode()[0]
    row = {"Mode": m, "Orders": len(sub), "Current promise": cur,
           "Current late %": (sub[REAL] > sub[SCHED]).mean() * 100}
    for q in (0.75, 0.90, 0.95):
        p = np.ceil(sub[REAL].quantile(q))
        row[f"P{int(q*100)} promise"] = p
        row[f"P{int(q*100)} late %"] = (sub[REAL] > p).mean() * 100
    opt_rows.append(row)
opt = pd.DataFrame(opt_rows).set_index("Mode").round(1)
print(opt.to_string())
tot = opt["Orders"].sum()
w = lambda col: (opt[col] * opt["Orders"]).sum() / tot      # order-weighted average
trade = pd.DataFrame({
    "Average promised days": [w("Current promise"), w("P75 promise"), w("P90 promise"), w("P95 promise")],
    "Overall late %": [w("Current late %"), w("P75 late %"), w("P90 late %"), w("P95 late %")]},
    index=["Current promises", "75th percentile", "90th percentile", "95th percentile"]).round(2)
print("\nTrade-off between promised time and lateness (all modes together):")
print(trade.to_string())
print("A longer promise means fewer late deliveries but a slower-looking service.")
cur_all, p90_all = trade.loc["Current promises", "Overall late %"], trade.loc["90th percentile", "Overall late %"]
summary["late_current"], summary["late_p90"] = cur_all, p90_all
summary["tradeoff"] = trade

plt.figure(figsize=(7, 4.5))
x = np.arange(len(opt))
bw = 0.27
plt.bar(x - bw, opt["Current late %"], bw, label="Current promise", color="#c0392b")
plt.bar(x, opt["P75 late %"], bw, label="75th percentile promise", color="#e9c46a")
plt.bar(x + bw, opt["P90 late %"], bw, label="90th percentile promise", color="#2a9d8f")
plt.xticks(x, opt.index); plt.ylabel("Late deliveries (%)")
plt.title("Late rate under current and optimised promised times"); plt.legend()
save("model5_promise_optimisation.png")


# ================================================================ PART B
title("PART B: FORECASTING DAILY ORDER VOLUME")
daily = df.groupby(df[DATE].dt.normalize()).size().rename("orders").to_frame()
daily = daily.asfreq("D", fill_value=0)

# the order volume drops suddenly in late 2017 (seen in Week 3); the forecast is trained
# on the period before the drop so that the model does not learn a pattern that may be a data gap
monthly = daily["orders"].resample("MS").sum()
normal = monthly.iloc[:12].median()
low = monthly[monthly < 0.6 * normal]
if len(low):
    cutoff = low.index[0]
    daily = daily[daily.index < cutoff]
    print("Sudden drop in volume detected. Using data before", cutoff.date())
else:
    print("No sudden drop detected. Using all data.")
print("Days used:", len(daily), "| average orders per day:", round(daily["orders"].mean(), 1))

f = daily.copy()
f["dow"] = f.index.dayofweek
f["month"] = f.index.month
f["day_of_month"] = f.index.day
f["lag_1"] = f["orders"].shift(1)
f["lag_7"] = f["orders"].shift(7)
f["lag_14"] = f["orders"].shift(14)
f["roll_7"] = f["orders"].shift(1).rolling(7).mean()
f["roll_28"] = f["orders"].shift(1).rolling(28).mean()
f["trend"] = np.arange(len(f))
f = f.dropna()
fcols = ["dow", "month", "day_of_month", "lag_1", "lag_7", "lag_14", "roll_7", "roll_28", "trend"]

n_test = 90
train_f, test_f = f.iloc[:-n_test], f.iloc[-n_test:]
print(f"Training days: {len(train_f)} | Test days (last {n_test}): {len(test_f)}")

fmodels = {
    "Same day last week": None,
    "Linear regression": LinearRegression(),
    "Random forest": RandomForestRegressor(n_estimators=200, max_depth=8, min_samples_leaf=5,
                                           random_state=SEED, n_jobs=-1),
    "Gradient boosting": GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                                   random_state=SEED),
}

title("B1. TIME-SERIES CROSS-VALIDATION (5 splits, training data only)")
tscv = TimeSeriesSplit(n_splits=5)
cv_b = {}
for name, mdl in fmodels.items():
    if mdl is None:
        continue
    res = cross_validate(mdl, train_f[fcols], train_f["orders"], cv=tscv,
                         scoring={"rmse": "neg_root_mean_squared_error", "r2": "r2"})
    cv_b[name] = {"CV RMSE": -res["test_rmse"].mean(), "CV R2": res["test_r2"].mean()}
print(pd.DataFrame(cv_b).T.round(3).to_string())

title("B2. TEST RESULTS (last 90 days)")
rows_b, preds_b = {}, {}
for name, mdl in fmodels.items():
    if mdl is None:
        p = test_f["lag_7"].values
    else:
        mdl.fit(train_f[fcols], train_f["orders"])
        p = mdl.predict(test_f[fcols])
    preds_b[name] = p
    rows_b[name] = metrics(test_f["orders"].values, p, mape=True)
tb = pd.DataFrame(rows_b).T.round(3)
print(tb.to_string())
best_b = tb["RMSE"].idxmin()
print("\nBest forecasting model:", best_b)
summary["best_forecast"] = best_b
summary["forecast_table"] = tb

plt.figure(figsize=(9, 4.5))
plt.plot(test_f.index, test_f["orders"], label="Actual", color="#1d3557")
plt.plot(test_f.index, preds_b[best_b], label="Forecast: " + best_b, color="#e63946", linestyle="--")
plt.ylabel("Order items per day"); plt.title("Daily order volume: actual vs forecast (last 90 days)")
plt.legend()
save("model6_demand_forecast.png")

# ---------------------------------------------------- optimisation: capacity planning
title("B3. OPTIMISATION: PLANNING VEHICLE CAPACITY FROM THE FORECAST")
print(f"Assumption: one vehicle can deliver {DELIVERIES_PER_VEHICLE_DAY} items per day (example value).")
rmse_b = tb.loc[best_b, "RMSE"]
pred = preds_b[best_b]
actual = test_f["orders"].values
need = lambda v: np.ceil(np.asarray(v) / DELIVERIES_PER_VEHICLE_DAY)

peak_plan = need(train_f["orders"].max())                  # same fleet every day, sized for the busiest day
fc_plan = need(pred + 1.28 * rmse_b)                       # forecast plus a safety margin (about 90% cover)
fc_plan_nobuf = need(pred)                                 # forecast only
for label, plan in [("Fixed fleet sized for the busiest day", np.full(len(actual), peak_plan)),
                    ("Forecast only, no safety margin", fc_plan_nobuf),
                    ("Forecast + safety margin (1.28 x RMSE)", fc_plan)]:
    short = int((plan * DELIVERIES_PER_VEHICLE_DAY < actual).sum())
    print(f"{label:42s} vehicle-days: {int(plan.sum()):5d} | days short of capacity: {short}")
vd_peak = int(np.full(len(actual), peak_plan).sum()); vd_fc = int(fc_plan.sum())
print(f"\nVehicle-days saved by forecast-based planning: {vd_peak - vd_fc} "
      f"({100 * (vd_peak - vd_fc) / vd_peak:.1f}%) over {len(actual)} days")
summary["vd_saved_pct"] = 100 * (vd_peak - vd_fc) / vd_peak


# ================================================================ summary
title("FINAL SUMMARY  (copy everything from here into the chat)")
print("Part A target:", REAL, "| inputs:", features)
print("Rows used (canceled removed):", len(d), "| train/test:", len(X_train), "/", len(X_test))
print("Tuned settings:", summary["tree_params"], summary["rf_params"])
print("\nCross-validation:\n", cv_table.to_string())
print("\nTest results:\n", test_table.to_string())
print("Best model:", summary["best_model"], "| most important input:", summary["top_feature"])
print("\nPromise trade-off:\n", summary["tradeoff"].to_string())
print("\nPart B forecast results:\n", summary["forecast_table"].to_string())
print("Best forecast model:", summary["best_forecast"])
print(f"Vehicle-days saved vs fixed peak fleet: {summary['vd_saved_pct']:.1f}%")
print("\nDone. Download the model1 to model6 PNG files.")
