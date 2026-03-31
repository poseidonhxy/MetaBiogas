#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import os
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib

# ========= config =========
INPUT_XLSX = "/Users/huangxiaoyan/Desktop/MetaBiogas/Clustering/Distance_A abundance/Raw data/All_correlations_with_MAG_Dist_PhylomInt.xlsx"
SHEET_NAME = "Sheet1"

OUT_DIR = "/Users/huangxiaoyan/Desktop/MetaBiogas/Clustering/Distance_A abundance/RandomForest_multi"
os.makedirs(OUT_DIR, exist_ok=True)

FEATURE_COLS = [
    "distance",
    "1 to 2 Competition",
    "2 to 1 Competition",
    "1 to 2 Complementarity",
    "2 to 1 Complementarity",
]
TARGET_COL = "Pearson_r"


def bin_strategy_qcut3(series: pd.Series):
    """
    S1 3 disturibution

        y (np.ndarray, int labels 0/1/2),
        bins_info (str),
        label_names (list[str]),
        class_counts (Series)
    """
    y_bin, bins = pd.qcut(
        series,
        q=3,
        labels=[0, 1, 2],
        retbins=True,
        duplicates="drop",
    )
    y_bin = y_bin.astype(int)

    label_names = ["Low", "Mid", "High"]
    bins_info_lines = ["=== Pearson_r  (q=3) ==="]
    for i in range(len(bins) - 1):
        bins_info_lines.append(
            f"  Bin {i} (label={i}, {label_names[i]}): [{bins[i]:.6f}, {bins[i+1]:.6f}]"
        )
    bins_info = "\n".join(bins_info_lines)

    class_counts = pd.Series(y_bin).value_counts().sort_index()
    return y_bin.values, bins_info, label_names, class_counts


def bin_strategy_fixed3(series: pd.Series):
    """
    S2：fix 3
      0: [-1, -0.5]
      1: (-0.5, 0.5]
      2: (0.5, 1]
    """
    # 确保 Pearson_r 在[-1,1] 范围内（理论上相关系数就应该如此）
    s = series.clip(-1, 1)

    bins = [-1.0, -0.5, 0.5, 1.0]
    labels = [0, 1, 2]
    y_bin = pd.cut(
        s,
        bins=bins,
        labels=labels,
        include_lowest=True,  # 包括最左边界 -1
        right=True,           # 区间为 (-1, -0.5], (-0.5, 0.5], (0.5, 1]
    ).astype(int)

    label_names = [
        "[-1, -0.5]",
        "(-0.5, 0.5]",
        "(0.5, 1]",
    ]

    bins_info = (
        "=== Pearson_r  ===\n"
        "  label 0: [-1, -0.5]\n"
        "  label 1: (-0.5, 0.5]\n"
        "  label 2: (0.5, 1]\n"
    )

    class_counts = y_bin.value_counts().sort_index()
    return y_bin.values, bins_info, label_names, class_counts


def bin_strategy_fixed2(series: pd.Series):
    """
    Fix 2 level
      0: [-1, 0]
      1: (0, 1]
    """
    s = series.clip(-1, 1)

    bins = [-1.0, 0.0, 1.0]
    labels = [0, 1]
    y_bin = pd.cut(
        s,
        bins=bins,
        labels=labels,
        include_lowest=True,
        right=True,
    ).astype(int)

    label_names = [
        "[-1, 0]",
        "(0, 1]",
    ]

    bins_info = (
        "=== Pearson_r  ===\n"
        "  label 0: [-1, 0]\n"
        "  label 1: (0, 1]\n"
    )

    class_counts = y_bin.value_counts().sort_index()
    return y_bin.values, bins_info, label_names, class_counts


def run_one_strategy(
    df_model: pd.DataFrame,
    strategy_name: str,
    bin_func,
):

    print("\n" + "=" * 80)
    print(f": {strategy_name}")
    print("=" * 80)

    y, bins_info, label_names, class_counts = bin_func(df_model[TARGET_COL])

    n_classes = len(np.unique(y))
    print(bins_info)
    print("\nsample number:")
    print(class_counts)

    min_count = class_counts.min()

    n_splits = min(5, int(min_count))
    if n_splits < 2:
        n_splits = 2

    X = df_model[FEATURE_COLS].values

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )
    print(f"train: {X_train.shape[0]}  test: {X_test.shape[0]} ")

    # RandomForest + grid search
    rf = RandomForestClassifier(
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    param_grid = {
        "n_estimators": [200, 500],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
        "max_features": ["sqrt", "log2"],
    }

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    grid = GridSearchCV(
        estimator=rf,
        param_grid=param_grid,
        cv=cv,
        scoring="f1_macro",
        n_jobs=-1,
        verbose=2,
    )

    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_
    print("best hyper:", grid.best_params_)
    print("CV  macro-F1:", f"{grid.best_score_:.4f}")

    # 测试集评估
    y_pred = best_rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    cls_report = classification_report(
        y_test,
        y_pred,
        digits=4,
        target_names=label_names,
    )
    cm = confusion_matrix(y_test, y_pred)

    print("Accuracy:", f"{acc:.4f}")
    print("\nClassification report:")
    print(cls_report)
    print(cm)

    # 特征重要性
    importances = best_rf.feature_importances_
    feat_imp_df = pd.DataFrame({
        "feature": FEATURE_COLS,
        "importance": importances,
    }).sort_values("importance", ascending=False)

    print(feat_imp_df)

    strategy_dir = os.path.join(OUT_DIR, strategy_name)
    os.makedirs(strategy_dir, exist_ok=True)

    summary_txt = os.path.join(strategy_dir, "rf_results_summary.txt")
    feat_imp_csv = os.path.join(strategy_dir, "rf_feature_importances.csv")
    best_model_pkl = os.path.join(strategy_dir, "rf_best_model.pkl")

    lines = []
    lines.append(f"=== Strategy: {strategy_name} ===\n\n")
    lines.append(f"Input file: {INPUT_XLSX}\n")
    lines.append("Feature columns:\n  " + ", ".join(FEATURE_COLS) + "\n\n")

    lines.append("=== Binning Info ===\n")
    lines.append(bins_info + "\n\n")

    lines.append("Class counts:\n")
    lines.append(str(class_counts) + "\n\n")

    lines.append("=== Best CV Result ===\n")
    lines.append("Best params:\n")
    lines.append(str(grid.best_params_) + "\n")
    lines.append(f"Best CV macro-F1: {grid.best_score_:.4f}\n\n")

    lines.append("=== Test Set Performance ===\n")
    lines.append(f"Accuracy: {acc:.4f}\n\n")
    lines.append("Classification report:\n")
    lines.append(cls_report + "\n")
    lines.append("Confusion matrix (rows=true, cols=pred):\n")
    lines.append(np.array2string(cm) + "\n\n")

    lines.append("=== Feature Importances ===\n")
    lines.append(feat_imp_df.to_string(index=False) + "\n")

    with open(summary_txt, "w", encoding="utf-8") as f:
        f.writelines(lines)


    feat_imp_df.to_csv(feat_imp_csv, index=False)
    joblib.dump(best_rf, best_model_pkl)


def main():
    df = pd.read_excel(INPUT_XLSX, sheet_name=SHEET_NAME)
    missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in df.columns]
    if missing_cols:
        raise ValueError(f"check: {missing_cols}")

    df_model = df[FEATURE_COLS + [TARGET_COL]].copy()
    before = len(df_model)
    df_model = df_model.dropna()
    after = len(df_model)

    strategies = [
        ("qcut3_equal_freq", bin_strategy_qcut3),
        ("fixed3_-1_-0.5_0.5_1", bin_strategy_fixed3),
        ("fixed2_-1_0_1", bin_strategy_fixed2),
    ]

    for name, func in strategies:
        run_one_strategy(df_model, name, func)


if __name__ == "__main__":
    main()
