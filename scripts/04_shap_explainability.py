"""
Run locally AFTER 03_features_and_model.py -- uses the saved model (xgb_model.pkl)
and re-derives the same feature matrix to compute SHAP values.

Covers Objective 5: "Apply SHAP for per-alert and model-level explainability."

Produces:
1. Global feature importance (mean |SHAP value|) -- model-level explanation
2. A SHAP summary plot (beeswarm) -- shows direction of effect, not just magnitude
3. Per-alert explanations for a handful of individual flagged transactions --
   the kind of output an analyst would actually see when reviewing an alert
4. A comparison: does the ACH feature's SHAP direction match the EDA finding?
   (sanity check that the model's reasoning is consistent with the data, not
   just a black box producing the right number by coincidence)

Requires: pip install shap  (in addition to xgboost, pandas, numpy already installed)

Usage:
    python 04_shap_explainability.py
"""

import json
import pickle
import pandas as pd
import numpy as np

try:
    import shap
except ImportError:
    raise SystemExit("Install shap first:  pip install shap")

try:
    import xgboost as xgb
except ImportError:
    raise SystemExit("Install xgboost first:  pip install xgboost")

# ---- config ----
CLEANED_PATH = "HI-Small_cleaned.csv"
MODEL_PATH = "xgb_model.pkl"
TRUNCATE_DATE = "2022-09-11"  # must match 03_features_and_model.py
N_PER_ALERT_EXAMPLES = 5       # how many individual alerts to fully explain
SHAP_SAMPLE_SIZE = 50_000      # subsample for the summary plot -- full test set is fine for
                                # TreeExplainer speed-wise, but a sample keeps the plot readable


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must exactly match the feature engineering in 03_features_and_model.py."""
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

    print("Computing 24h account velocity feature (can take ~1 min)...")
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
    print(f"Loading model from {MODEL_PATH} ...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    print(f"Loading {CLEANED_PATH} ...")
    df = pd.read_csv(CLEANED_PATH, parse_dates=['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df = df[df['Timestamp'] < pd.Timestamp(TRUNCATE_DATE)].reset_index(drop=True)
    print(f"Rows after truncation: {len(df):,}")

    y = df['Is Laundering']
    X = engineer_features(df)

    # recreate the same temporal test split
    n = len(df)
    val_end = int(n * 0.85)
    X_test = X.iloc[val_end:].reset_index(drop=True)
    y_test = y.iloc[val_end:].reset_index(drop=True)
    df_test = df.iloc[val_end:].reset_index(drop=True)
    print(f"Test set: {len(X_test):,} rows, {int(y_test.sum())} laundering")

    # --- SHAP values via TreeExplainer (fast, exact for tree models) ---
    print("\nComputing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)

    sample_idx = np.random.RandomState(42).choice(
        len(X_test), size=min(SHAP_SAMPLE_SIZE, len(X_test)), replace=False
    )
    X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
    shap_values = explainer.shap_values(X_sample)

    # --- 1. global feature importance (model-level explanation) ---
    mean_abs_shap = pd.Series(
        np.abs(shap_values).mean(axis=0), index=X.columns
    ).sort_values(ascending=False)
    print("\nTop 10 features by mean |SHAP value| (global importance):")
    print(mean_abs_shap.head(10).to_string())

    # --- 2. summary plot (beeswarm) ---
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved -> shap_summary_plot.png")

    # bar version (cleaner for a report figure)
    shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False, max_display=15)
    plt.tight_layout()
    plt.savefig('shap_summary_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved -> shap_summary_bar.png")

    # --- 3. per-alert explanations: a few individual flagged transactions ---
    y_proba_sample = model.predict_proba(X_sample)[:, 1]
    flagged_idx = np.where(y_proba_sample >= 0.5)[0]
    print(f"\n{len(flagged_idx)} alerts in this sample at default threshold.")

    per_alert_examples = []
    example_idx = flagged_idx[:N_PER_ALERT_EXAMPLES] if len(flagged_idx) >= N_PER_ALERT_EXAMPLES else flagged_idx
    for idx in example_idx:
        row_shap = pd.Series(shap_values[idx], index=X.columns).sort_values(key=abs, ascending=False)
        top_contributors = row_shap.head(5)
        per_alert_examples.append({
            "row_probability": round(float(y_proba_sample[idx]), 4),
            "actual_is_laundering": int(y_test.iloc[sample_idx[idx]]),
            "top_5_contributing_features": {
                k: round(float(v), 4) for k, v in top_contributors.items()
            },
        })

    print("\nExample per-alert explanations (top 5 contributing features each):")
    for i, ex in enumerate(per_alert_examples):
        print(f"\nAlert {i+1}: probability={ex['row_probability']}, actually laundering={bool(ex['actual_is_laundering'])}")
        for feat_name, val in ex['top_5_contributing_features'].items():
            direction = "pushes toward alert" if val > 0 else "pushes away from alert"
            print(f"    {feat_name}: {val:+.4f}  ({direction})")

    # --- 4. sanity check: does ACH's SHAP direction match the EDA finding? ---
    ach_shap = shap_values[:, X.columns.get_loc('is_ach')]
    ach_present_mean_shap = ach_shap[X_sample['is_ach'] == 1].mean() if (X_sample['is_ach'] == 1).any() else None
    ach_absent_mean_shap = ach_shap[X_sample['is_ach'] == 0].mean() if (X_sample['is_ach'] == 0).any() else None
    print(f"\nSanity check -- mean SHAP value for is_ach:")
    print(f"  When ACH=1: {ach_present_mean_shap:.4f}  (should be positive -- pushes toward laundering)")
    print(f"  When ACH=0: {ach_absent_mean_shap:.4f}  (should be near zero or negative)")

    summary = {
        "n_sample": len(X_sample),
        "n_laundering_in_sample": int(y_test.iloc[sample_idx].sum()),
        "top_10_global_shap_importance": mean_abs_shap.head(10).to_dict(),
        "ach_shap_sanity_check": {
            "mean_shap_when_ach_true": round(float(ach_present_mean_shap), 4) if ach_present_mean_shap is not None else None,
            "mean_shap_when_ach_false": round(float(ach_absent_mean_shap), 4) if ach_absent_mean_shap is not None else None,
        },
        "n_alerts_in_sample_default_threshold": int(len(flagged_idx)),
        "per_alert_examples": per_alert_examples,
        "output_files": ["shap_summary_plot.png", "shap_summary_bar.png"],
    }

    print("\n" + "=" * 60)
    print("===SUMMARY_START===")
    print(json.dumps(summary, indent=2, default=str))
    print("===SUMMARY_END===")
    print("=" * 60)
    print("\nCopy everything between the SUMMARY_START / SUMMARY_END markers and send it back.")
    print("Also send over shap_summary_plot.png and shap_summary_bar.png if you can (or describe what you see).")


if __name__ == "__main__":
    main()
