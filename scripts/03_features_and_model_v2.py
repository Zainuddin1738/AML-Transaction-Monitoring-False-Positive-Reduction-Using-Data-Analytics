"""
v2 -- addresses gaps found on review before moving to SHAP:
1. Early stopping now actually active (early_stopping_rounds set -- v1 tracked the
   validation set but never used it to stop training or select the best round).
2. Train-set metrics reported alongside test-set metrics, to check for overfitting.
3. A second, simpler classifier added (cost-sensitive Logistic Regression) --
   Objective 4 says "classifier(s)", and this justifies XGBoost's added complexity
   by showing what a simpler linear model achieves for comparison.
4. Full precision-recall curve data exported for plotting (previously only 3
   discrete threshold points were reported).

Requires: pip install xgboost scikit-learn pandas numpy

Usage:
    python 03_features_and_model_v2.py
"""

import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score, precision_score, recall_score,
    confusion_matrix
)
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
except ImportError:
    raise SystemExit("Install xgboost first:  pip install xgboost")

# ---- config ----
CLEANED_PATH = "HI-Small_cleaned.csv"
MODEL_OUT_PATH = "xgb_model.pkl"
LR_MODEL_OUT_PATH = "lr_model.pkl"
PREDICTIONS_OUT_PATH = "ml_predictions.csv"
PR_CURVE_OUT_PATH = "pr_curve_data.csv"

TRUNCATE_DATE = "2022-09-11"  # see 01_data_cleaning.ipynb for rationale

BASELINE_RECALL = 0.907121
BASELINE_ALERTS = 2_607_758
BASELINE_N_ROWS = 5_077_237

RANDOM_STATE = 42


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    feat = pd.DataFrame(index=df.index)
    feat['amount_paid'] = df['Amount Paid']
    feat['log_amount_paid'] = np.log1p(df['Amount Paid'])
    feat['above_ctr_threshold'] = (df['Amount Paid'] > 10_000).astype(int)
    feat['in_structuring_band'] = ((df['Amount Paid'] >= 9_000) & (df['Amount Paid'] < 10_000)).astype(int)
    feat['is_round_amount'] = (df['Amount Paid'] % 1_000 == 0).astype(int)
    feat['currency_mismatch'] = df['currency_mismatch']
    feat['is_ach'] = (df['Payment Format'] == 'ACH').astype(int)
    feat['is_bitcoin'] = (df['Payment Format'] == 'Bitcoin').astype(int)
    feat['is_cash'] = (df['Payment Format'] == 'Cash').astype(int)
    fmt_dummies = pd.get_dummies(df['Payment Format'], prefix='fmt', dtype=int)
    feat = pd.concat([feat, fmt_dummies], axis=1)
    feat['hour'] = df['hour']
    feat['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    dow_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    feat['day_of_week_num'] = df['day_of_week'].map(dow_map)
    feat['is_self_transaction'] = (
        (df['From Account'] == df['To Account']) & (df['From Bank'] == df['To Bank'])
    ).astype(int)

    print("Computing 24h account velocity feature (can take ~1 min on full data)...")
    tmp = df[['From Account', 'Timestamp']].copy()
    tmp['ones'] = 1
    tmp = tmp.sort_values(['From Account', 'Timestamp'])
    rolling_counts = (
        tmp.set_index('Timestamp')
           .groupby('From Account')['ones']
           .rolling('24h')
           .sum()
           .reset_index(drop=True)
    )
    tmp = tmp.reset_index(drop=True)
    tmp['velocity_24h'] = rolling_counts.values
    velocity_result = pd.Series(0.0, index=df.index)
    velocity_result.iloc[tmp.index.values] = tmp['velocity_24h'].values
    feat['velocity_24h'] = velocity_result

    return feat


def evaluate_at_threshold(y_true, y_proba, threshold):
    y_pred = (y_proba >= threshold).astype(int)
    return {
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 6),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 6),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 6),
        "n_alerts": int(y_pred.sum()),
    }


def main():
    print(f"Loading {CLEANED_PATH} ...")
    df = pd.read_csv(CLEANED_PATH, parse_dates=['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)

    n_before = len(df)
    df = df[df['Timestamp'] < pd.Timestamp(TRUNCATE_DATE)].reset_index(drop=True)
    print(f"Truncated: {n_before:,} -> {len(df):,} rows")

    y = df['Is Laundering']
    X = engineer_features(df)
    print(f"Feature matrix shape: {X.shape}")

    n = len(df)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    print(f"\nTrain: {len(X_train):,} rows, {y_train.sum()} laundering")
    print(f"Val:   {len(X_val):,} rows, {y_val.sum()} laundering")
    print(f"Test:  {len(X_test):,} rows, {y_test.sum()} laundering")

    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"\nscale_pos_weight: {scale_pos_weight:.2f}")

    # ================= MODEL 1: XGBoost, with early stopping now actually active =================
    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
        early_stopping_rounds=30,   # <-- the actual fix: v1 set eval_set but not this
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    print("\nTraining XGBoost (with early stopping)...")
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    best_iteration = model.best_iteration
    print(f"Training complete. Best iteration: {best_iteration} (of 300 max)")

    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_test = model.predict_proba(X_test)[:, 1]

    train_metrics_default = evaluate_at_threshold(y_train, y_proba_train, 0.5)
    test_metrics_default = evaluate_at_threshold(y_test, y_proba_test, 0.5)

    print("\n--- Overfitting check: train vs. test at default threshold ---")
    print(f"Train: precision={train_metrics_default['precision']:.4f}, recall={train_metrics_default['recall']:.4f}, f1={train_metrics_default['f1']:.4f}")
    print(f"Test:  precision={test_metrics_default['precision']:.4f}, recall={test_metrics_default['recall']:.4f}, f1={test_metrics_default['f1']:.4f}")
    overfit_gap_f1 = train_metrics_default['f1'] - test_metrics_default['f1']
    print(f"F1 gap (train - test): {overfit_gap_f1:.4f}  (large gap = overfitting concern)")

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba_test)
    pr_auc = auc(recalls, precisions)

    # export full PR curve for plotting
    pr_curve_df = pd.DataFrame({
        'threshold': list(thresholds) + [1.0],
        'precision': precisions,
        'recall': recalls,
    })
    pr_curve_df.to_csv(PR_CURVE_OUT_PATH, index=False)
    print(f"\nSaved PR curve data -> {PR_CURVE_OUT_PATH}")

    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores[:-1])
    best_f1_threshold = thresholds[best_f1_idx] if len(thresholds) > 0 else 0.5

    target_recall = BASELINE_RECALL
    valid_idx = np.where(recalls[:-1] >= target_recall)[0]
    if len(valid_idx) > 0:
        best_idx = valid_idx[np.argmax(precisions[valid_idx])]
        matched_threshold = thresholds[best_idx]
        matched_precision = precisions[best_idx]
        matched_recall = recalls[best_idx]
        n_alerts_matched = int((y_proba_test >= matched_threshold).sum())
        test_fraction = len(X_test) / BASELINE_N_ROWS
        n_alerts_matched_scaled = int(n_alerts_matched / test_fraction)
    else:
        matched_threshold = matched_precision = matched_recall = None
        n_alerts_matched = n_alerts_matched_scaled = None

    importance = dict(zip(X.columns, model.feature_importances_.tolist()))
    top_features_xgb = dict(sorted(importance.items(), key=lambda x: -x[1])[:10])

    # ================= MODEL 2: Cost-sensitive Logistic Regression (comparison) =================
    print("\nTraining Logistic Regression (class_weight='balanced') for comparison...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr_model = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=RANDOM_STATE)
    lr_model.fit(X_train_scaled, y_train)
    y_proba_lr_test = lr_model.predict_proba(X_test_scaled)[:, 1]

    lr_precisions, lr_recalls, _ = precision_recall_curve(y_test, y_proba_lr_test)
    lr_pr_auc = auc(lr_recalls, lr_precisions)
    lr_metrics_default = evaluate_at_threshold(y_test, y_proba_lr_test, 0.5)

    print(f"Logistic Regression -- PR-AUC: {lr_pr_auc:.4f}, default-threshold: "
          f"precision={lr_metrics_default['precision']:.4f}, recall={lr_metrics_default['recall']:.4f}")
    print(f"XGBoost              -- PR-AUC: {pr_auc:.4f}, default-threshold: "
          f"precision={test_metrics_default['precision']:.4f}, recall={test_metrics_default['recall']:.4f}")

    lr_coef = dict(zip(X.columns, lr_model.coef_[0].tolist()))
    top_lr_coef = dict(sorted(lr_coef.items(), key=lambda x: -abs(x[1]))[:10])

    # ================= save models and predictions =================
    with open(MODEL_OUT_PATH, 'wb') as f:
        pickle.dump(model, f)
    with open(LR_MODEL_OUT_PATH, 'wb') as f:
        pickle.dump({'model': lr_model, 'scaler': scaler}, f)
    print(f"\nSaved models -> {MODEL_OUT_PATH}, {LR_MODEL_OUT_PATH}")

    test_out = df.iloc[val_end:][['Timestamp', 'From Account', 'To Account', 'Amount Paid', 'Is Laundering']].copy()
    test_out['ml_probability'] = y_proba_test
    test_out['ml_alert_default_threshold'] = (y_proba_test >= 0.5).astype(int)
    test_out['lr_probability'] = y_proba_lr_test
    if matched_threshold is not None:
        test_out['ml_alert_matched_recall'] = (y_proba_test >= matched_threshold).astype(int)
    test_out.to_csv(PREDICTIONS_OUT_PATH, index=False)
    print(f"Saved test-set predictions -> {PREDICTIONS_OUT_PATH}")

    summary = {
        "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
        "n_laundering_train": int(y_train.sum()), "n_laundering_val": int(y_val.sum()), "n_laundering_test": int(y_test.sum()),
        "scale_pos_weight": round(float(scale_pos_weight), 2),
        "xgboost": {
            "best_iteration": int(best_iteration),
            "pr_auc": round(float(pr_auc), 6),
            "train_default_threshold": train_metrics_default,
            "test_default_threshold": test_metrics_default,
            "overfit_f1_gap": round(float(overfit_gap_f1), 6),
            "best_f1_threshold": round(float(best_f1_threshold), 4),
            "matched_to_baseline_recall": {
                "baseline_recall": BASELINE_RECALL,
                "ml_threshold_used": round(float(matched_threshold), 4) if matched_threshold is not None else None,
                "ml_precision_at_this_recall": round(float(matched_precision), 6) if matched_precision is not None else None,
                "ml_recall_achieved": round(float(matched_recall), 6) if matched_recall is not None else None,
                "ml_n_alerts_scaled_to_full_dataset": n_alerts_matched_scaled,
            },
            "top_10_feature_importance": top_features_xgb,
        },
        "logistic_regression": {
            "pr_auc": round(float(lr_pr_auc), 6),
            "test_default_threshold": lr_metrics_default,
            "top_10_abs_coefficients": top_lr_coef,
        },
        "model_comparison_note": "XGBoost vs. Logistic Regression PR-AUC ratio: "
                                  f"{round(pr_auc / lr_pr_auc, 2) if lr_pr_auc > 0 else None}x",
    }

    print("\n" + "=" * 60)
    print("===SUMMARY_START===")
    print(json.dumps(summary, indent=2, default=str))
    print("===SUMMARY_END===")
    print("=" * 60)
    print("\nCopy everything between the SUMMARY_START / SUMMARY_END markers and send it back.")
    print("Also send pr_curve_data.csv if you can (or just the summary is fine).")


if __name__ == "__main__":
    main()
