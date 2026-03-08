#!/usr/bin/env python
# coding: utf-8

import os
import joblib
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import numpy as np
import pandas as pd

from sklearn import datasets
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

RANDOM_STATE = 42

# ======================
# Data Loading
# ======================
raw = datasets.fetch_california_housing()
df = pd.DataFrame(raw.data, columns=raw.feature_names)
df["Price"] = raw.target

# ======================
# Cleaning
# ======================
mask = (
    (df["AveRooms"] < 10) &
    (df["AveBedrms"] < 10) &
    (df["Population"] < 15000) &
    (df["AveOccup"] < 10) &
    (df["Price"] < 5)
)
df = df.loc[mask].copy()

# ======================
# Feature Engineering
# ======================
df["RoomsPerPerson"] = df["AveRooms"] / df["AveOccup"]
df["BedroomsRatio"] = df["AveBedrms"] / df["AveRooms"]
df["PopulationDensity"] = df["Population"] / df["HouseAge"]

TARGET = "Price"
FEATURES = [c for c in df.columns if c != TARGET]

X = df[FEATURES]
y = df[TARGET]

# ======================
# Split & Scaling
# ======================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

# ======================
# Random Forest
# ======================
rf_param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [8, 16, None],
    "min_samples_split": [2, 5],
    "max_features": ["sqrt", 0.5],
}

rf = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1)
rf_grid = GridSearchCV(rf, rf_param_grid, cv=3,
                       scoring="neg_root_mean_squared_error",
                       n_jobs=-1)
rf_grid.fit(X_train_sc, y_train)

# ======================
# XGBoost
# ======================
xgb_param_grid = {
    "n_estimators": [200, 400],
    "max_depth": [4, 6],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

xgb = XGBRegressor(objective="reg:squarederror",
                   random_state=RANDOM_STATE,
                   n_jobs=-1, verbosity=0)

xgb_grid = GridSearchCV(xgb, xgb_param_grid, cv=3,
                        scoring="neg_root_mean_squared_error",
                        n_jobs=-1)
xgb_grid.fit(X_train_sc, y_train)

# ======================
# Evaluation
# ======================
best_rf = rf_grid.best_estimator_
best_xgb = xgb_grid.best_estimator_

def evaluate(model):
    pred = model.predict(X_test_sc)
    return (
        np.sqrt(mean_squared_error(y_test, pred)),
        mean_absolute_error(y_test, pred),
        r2_score(y_test, pred)
    )

rmse_rf, mae_rf, r2_rf = evaluate(best_rf)
rmse_xgb, mae_xgb, r2_xgb = evaluate(best_xgb)

if rmse_xgb < rmse_rf:
    best_model = best_xgb
    best_name = "XGBoost"
    rmse, mae, r2 = rmse_xgb, mae_xgb, r2_xgb
    best_params = xgb_grid.best_params_
else:
    best_model = best_rf
    best_name = "RandomForest"
    rmse, mae, r2 = rmse_rf, mae_rf, r2_rf
    best_params = rf_grid.best_params_

# ======================
# Save Artifacts
# ======================
os.makedirs("models", exist_ok=True)
joblib.dump(best_model, "models/model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(FEATURES, "models/features.pkl")

# ======================
# MLflow Logging
# ======================
mlflow.set_experiment("california-housing")

with mlflow.start_run(run_name=f"{best_name}-best"):
    mlflow.log_params(best_params)
    mlflow.log_metrics({"rmse": rmse, "mae": mae, "r2": r2})
    mlflow.log_artifacts("models")
    mlflow.sklearn.log_model(best_model, "model")

print(f"Training complete — Best model: {best_name} | R2: {r2:.3f}")