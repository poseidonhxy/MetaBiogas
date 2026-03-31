

import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
import joblib

INPUT_XLSX = "/Users/huangxiaoyan/Desktop/MetaBiogas/Clustering/Distance_A abundance/Raw data/All_correlations_with_MAG_Dist_PhylomInt.xlsx"
SHEET_NAME = "Sheet1"

OUT_DIR = "/Users/huangxiaoyan/Desktop/MetaBiogas/Clustering/Distance_A abundance/SVR_regression"
os.makedirs(OUT_DIR, exist_ok=True)

SUMMARY_TXT = os.path.join(OUT_DIR, "svr_results_summary.txt")
PRED_CSV    = os.path.join(OUT_DIR, "svr_predictions_test.csv")
BEST_MODEL_PKL = os.path.join(OUT_DIR, "svr_best_model.pkl")

FEATURE_COLS = [
    "distance",
    "1 to 2 Competition",
    "2 to 1 Competition",
    "1 to 2 Complementarity",
    "2 to 1 Complementarity",
]
TARGET_COL = "Pearson_r"


def main():
    print(f"file: {INPUT_XLSX}")
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
    print("row number:", len(df))

    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"missing cols: {missing_cols}")

    df_model = df[FEATURE_COLS + [TARGET_COL]].copy()
    before = len(df_model)
    df_model = df_model.dropna()
    after = len(df_model)

    X = df_model[FEATURE_COLS].values
    y = df_model[TARGET_COL].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svr", SVR(kernel="rbf"))
    ])

    param_grid = {
        "svr__C": [1, 10, 100, 1000],
        "svr__gamma": ["scale", 0.1, 0.01, 0.001],
        "svr__epsilon": [0.01, 0.05, 0.1, 0.2],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=pipe,
        param_grid=param_grid,
        cv=cv,
        scoring="r2",
        n_jobs=-1,
        verbose=2,
    )

    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_


    y_pred = best_model.predict(X_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    try:
        corr, corr_p = pearsonr(y_test, y_pred)
    except Exception:
        corr, corr_p = np.nan, np.nan

    print(f"R2   : {r2:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"RMSE : {rmse:.4f}")
    print(f"corr(y_true, y_pred) = {corr:.4f} (p={corr_p:.3e})")

    pred_df = pd.DataFrame({
        "y_true": y_test,
        "y_pred": y_pred,
    })
    pred_df.to_csv(PRED_CSV, index=False)
    print(f"\nprediction result path: {PRED_CSV}")

    lines = []
    lines.append("=== SVR Regression on Pearson_r ===\n\n")
    lines.append(f"Input file: {INPUT_XLSX}\n")
    lines.append("Feature columns:\n  " + ", ".join(FEATURE_COLS) + "\n\n")

    lines.append("=== Best CV Result ===\n")
    lines.append("Best params:\n")
    lines.append(str(grid.best_params_) + "\n")
    lines.append(f"Best CV R2: {grid.best_score_:.4f}\n\n")

    lines.append("=== Test Set Performance ===\n")
    lines.append(f"R2   : {r2:.4f}\n")
    lines.append(f"MAE  : {mae:.4f}\n")
    lines.append(f"RMSE : {rmse:.4f}\n")
    lines.append(f"corr(y_true, y_pred) = {corr:.4f} (p={corr_p:.3e})\n")

    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.writelines(lines)

    print(f"summary path: {SUMMARY_TXT}")

    joblib.dump(best_model, BEST_MODEL_PKL)
    print(f"best model: {BEST_MODEL_PKL}")


if __name__ == "__main__":
    main()
