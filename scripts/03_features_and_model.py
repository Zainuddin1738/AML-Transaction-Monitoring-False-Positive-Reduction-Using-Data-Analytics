"""
Run locally AFTER 01_clean_data_full.py, on the full cleaned dataset.

Builds features, trains a cost-sensitive XGBoost classifier, evaluates it,
and -- critically -- compares it against the rule-based baseline at MATCHED
RECALL, so you get a fair, direct "for the same detection rate, how many
fewer alerts does ML require" comparison. That comparison is the single
most persuasive result for your Objective 6 / dissertation results section.

Requires: pip install xgboost scikit-learn pandas numpy

Usage:
    python 03_features_and_model.py
"""

import json
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import (
    precision_recall_curve, auc, f1_score, precision_score, recall_score,
    confusion_matrix
)

try:
    import xgboost as xgb
except ImportError:
    raise SystemExit("Install xgboost first:  pip install xgboost")

# ---- config ----
CLEANED_PATH = "HI-Small_cleaned.csv"
MODEL_OUT_PATH = "xgb_model.pkl"
PREDICTIONS_OUT_PATH = "ml_predictions.csv"

# Recall the rule-based baseline achieved -- used for the matched-recall comparison.
# Corrected to the final trimmed-window baseline (02_rule_baseline_full_v2.py, run on the
# same Sep 1-10 stable window as this script).
BASELINE_RECALL = 0.907121
BASELINE_ALERTS = 2_607_758
BASELINE_N_ROWS = 5_077_237

RANDOM_STATE = 42

# The dataset's background transaction volume collapses after ~Sep 10 (confirmed via
# 01b_temporal_and_account_check.py: daily laundering rate jumps from ~0.1% to 57-73%
# because total daily transaction volume drops to a few hundred, dominated by residual
# laundering-pattern completions). This is a simulator artifact, not a real pattern --
# training/testing on it would give misleading results. We restrict to the stable window.
TRUNCATE_DATE = "2022-09-11"  # exclusive -- keep Sep 1 through Sep 10 only


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the model's feature set. Kept to simple, defensible, non-graph
    features (per the project's explicit scope: graph/network analytics
    was ruled out) -- amount, timing, format, and lightweight per-account
    velocity aggregates only.
    """
    feat = pd.DataFrame(index=df.index)

    # --- amount features ---
    feat['amount_paid'] = df['Amount Paid']
    feat['log_amount_paid'] = np.log1p(df['Amount Paid'])
    feat['above_ctr_threshold'] = (df['Amount Paid'] > 10_000).astype(int)
    feat['in_structuring_band'] = ((df['Amount Paid'] >= 9_000) & (df['Amount Paid'] < 10_000)).astype(int)
    feat['is_round_amount'] = (df['Amount Paid'] % 1_000 == 0).astype(int)

    # --- currency / format features ---
    feat['currency_mismatch'] = df['currency_mismatch']
    feat['is_ach'] = (df['Payment Format'] == 'ACH').astype(int)          # empirically the strongest single signal found
    feat['is_bitcoin'] = (df['Payment Format'] == 'Bitcoin').astype(int)
    feat['is_cash'] = (df['Payment Format'] == 'Cash').astype(int)
    # one-hot the rest of payment format for the model to use directly too
    fmt_dummies = pd.get_dummies(df['Payment Format'], prefix='fmt', dtype=int)
    feat = pd.concat([feat, fmt_dummies], axis=1)

    # --- timing features ---
    feat['hour'] = df['hour']
    feat['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    feat['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    dow_map = {'Monday': 0, 'Tuesday': 1, 'Wednesday': 2, 'Thursday': 3,
               'Friday': 4, 'Saturday': 5, 'Sunday': 6}
    feat['day_of_week_num'] = df['day_of_week'].map(dow_map)

    # --- self-transaction flag ---
    feat['is_self_transaction'] = (
        (df['From Account'] == df['To Account']) & (df['From Bank'] == df['To Bank'])
    ).astype(int)

    # --- lightweight per-account velocity feature (NOT full graph analytics --
    # a simple rolling count, consistent with project scope) ---
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


def main():
    print(f"Loading {CLEANED_PATH} ...")
    df = pd.read_csv(CLEANED_PATH, parse_dates=['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    print(f"Loaded shape: {df.shape}")

    # --- restrict to the stable window (see TRUNCATE_DATE comment above) ---
    n_before = len(df)
    df = df[df['Timestamp'] < pd.Timestamp(TRUNCATE_DATE)].reset_index(drop=True)
    n_after = len(df)
    print(f"Truncated to {TRUNCATE_DATE} (excl.): {n_before:,} -> {n_after:,} rows "
          f"({n_before - n_after:,} dropped from the degenerate tail)")
    print(f"Laundering cases remaining: {int(df['Is Laundering'].sum())}")

    y = df['Is Laundering']
    X = engineer_features(df)
    print(f"Feature matrix shape: {X.shape}")
    print(f"Features: {X.columns.tolist()}")

    # --- temporal split: 70% train / 15% val / 15% test, in time order ---
    n = len(df)
    train_end = int(n * 0.70)
    val_end = int(n * 0.85)

    X_train, y_train = X.iloc[:train_end], y.iloc[:train_end]
    X_val, y_val = X.iloc[train_end:val_end], y.iloc[train_end:val_end]
    X_test, y_test = X.iloc[val_end:], y.iloc[val_end:]

    print(f"\nTrain: {len(X_train):,} rows, {y_train.sum()} laundering")
    print(f"Val:   {len(X_val):,} rows, {y_val.sum()} laundering")
    print(f"Test:  {len(X_test):,} rows, {y_test.sum()} laundering")

    if y_train.sum() < 5 or y_test.sum() < 5:
        print("\nWARNING: very few positive cases in a split -- results will be unstable.")
        print("Consider a different split ratio or investigate temporal clustering of laundering cases.")

    # --- cost-sensitive weighting, calibrated to the REAL class ratio ---
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    print(f"\nscale_pos_weight (neg/pos ratio in train): {scale_pos_weight:.2f}")

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        eval_metric='aucpr',
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    print("\nTraining XGBoost classifier...")
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )
    print("Training complete.")

    # --- evaluate on test set ---
    y_proba = model.predict_proba(X_test)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc = auc(recalls, precisions)

    # default 0.5 threshold metrics
    y_pred_default = (y_proba >= 0.5).astype(int)
    cm_default = confusion_matrix(y_test, y_pred_default)
    tn, fp, fn, tp = cm_default.ravel()
    f1_default = f1_score(y_test, y_pred_default, zero_division=0)
    precision_default = precision_score(y_test, y_pred_default, zero_division=0)
    recall_default = recall_score(y_test, y_pred_default, zero_division=0)

    # threshold that maximizes F1
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_f1_idx = np.argmax(f1_scores[:-1])  # last point has no matching threshold
    best_f1_threshold = thresholds[best_f1_idx] if len(thresholds) > 0 else 0.5
    best_f1 = f1_scores[best_f1_idx]
    best_f1_precision = precisions[best_f1_idx]
    best_f1_recall = recalls[best_f1_idx]

    # --- THE KEY COMPARISON: at the SAME recall the rule baseline achieved,
    # how many alerts does the ML model need? ---
    target_recall = BASELINE_RECALL
    # find the threshold achieving closest recall >= target_recall
    valid_idx = np.where(recalls[:-1] >= target_recall)[0]
    if len(valid_idx) > 0:
        # among thresholds achieving at least target recall, pick the one with highest precision
        best_idx = valid_idx[np.argmax(precisions[valid_idx])]
        matched_threshold = thresholds[best_idx]
        matched_precision = precisions[best_idx]
        matched_recall = recalls[best_idx]
        y_pred_matched = (y_proba >= matched_threshold).astype(int)
        n_alerts_matched = int(y_pred_matched.sum())
        # scale to full-dataset-equivalent alert count for fair comparison to baseline
        test_fraction = len(X_test) / BASELINE_N_ROWS
        n_alerts_matched_scaled = int(n_alerts_matched / test_fraction) if test_fraction > 0 else None
    else:
        matched_threshold = matched_precision = matched_recall = None
        n_alerts_matched = n_alerts_matched_scaled = None
        print(f"\nWARNING: model could not reach baseline recall ({target_recall:.1%}) on the test set.")

    # --- feature importance (quick look, full SHAP comes in notebook 06) ---
    importance = dict(zip(X.columns, model.feature_importances_.tolist()))
    top_features = dict(sorted(importance.items(), key=lambda x: -x[1])[:10])

    # --- save model and predictions ---
    with open(MODEL_OUT_PATH, 'wb') as f:
        pickle.dump(model, f)
    print(f"\nSaved model -> {MODEL_OUT_PATH}")

    test_out = df.iloc[val_end:][['Timestamp', 'From Account', 'To Account', 'Amount Paid', 'Is Laundering']].copy()
    test_out['ml_probability'] = y_proba
    test_out['ml_alert_default_threshold'] = y_pred_default
    if matched_threshold is not None:
        test_out['ml_alert_matched_recall'] = y_pred_matched
    test_out.to_csv(PREDICTIONS_OUT_PATH, index=False)
    print(f"Saved test-set predictions -> {PREDICTIONS_OUT_PATH}")

    summary = {
        "n_train": int(len(X_train)), "n_val": int(len(X_val)), "n_test": int(len(X_test)),
        "n_laundering_train": int(y_train.sum()), "n_laundering_val": int(y_val.sum()), "n_laundering_test": int(y_test.sum()),
        "scale_pos_weight": round(float(scale_pos_weight), 2),
        "pr_auc": round(float(pr_auc), 6),
        "default_threshold_0.5": {
            "precision": round(float(precision_default), 6),
            "recall": round(float(recall_default), 6),
            "f1": round(float(f1_default), 6),
            "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        },
        "best_f1_threshold": {
            "threshold": round(float(best_f1_threshold), 4),
            "precision": round(float(best_f1_precision), 6),
            "recall": round(float(best_f1_recall), 6),
            "f1": round(float(best_f1), 6),
        },
        "matched_to_baseline_recall": {
            "baseline_recall": BASELINE_RECALL,
            "baseline_alerts_full_dataset": BASELINE_ALERTS,
            "ml_threshold_used": round(float(matched_threshold), 4) if matched_threshold is not None else None,
            "ml_precision_at_this_recall": round(float(matched_precision), 6) if matched_precision is not None else None,
            "ml_recall_achieved": round(float(matched_recall), 6) if matched_recall is not None else None,
            "ml_n_alerts_on_test_set": n_alerts_matched,
            "ml_n_alerts_scaled_to_full_dataset_equivalent": n_alerts_matched_scaled,
        },
        "top_10_feature_importance": top_features,
    }

    print("\n" + "=" * 60)
    print("===SUMMARY_START===")
    print(json.dumps(summary, indent=2, default=str))
    print("===SUMMARY_END===")
    print("=" * 60)
    print("\nCopy everything between the SUMMARY_START / SUMMARY_END markers and send it back.")


if __name__ == "__main__":
    main()
