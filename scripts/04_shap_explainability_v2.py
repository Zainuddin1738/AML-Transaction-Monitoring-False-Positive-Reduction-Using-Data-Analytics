"""
v2 -- refines 04_shap_explainability.py based on reviewing its first output:

1. Guarantees ALL test-set laundering cases are included in the SHAP sample (not a
   pure random sample, which under-represents the rare positive class -- the first
   run's random 50K sample only had 69 of 906 laundering cases, too few non-ACH
   examples to test the hypothesis below).
2. Pulls STRATIFIED per-alert examples: true positives, false positives, AND false
   negatives -- the first run's 5 examples were all false positives, which is not a
   representative or persuasive set for an analyst-trust narrative.
3. Directly tests the hypothesis raised by 05_cost_based_evaluation.py's finding
   (ML's false negatives are ~24x more costly on average than the baseline's):
   is amount_paid's SHAP contribution specifically suppressed for non-ACH
   transactions? This is checked directly rather than left as a plausible guess.
4. Saves SHAP values to disk so this doesn't need recomputing again.

Usage:
    python 04_shap_explainability_v2.py
"""

import json
import pickle
import pandas as pd
import numpy as np

import shap
import xgboost as xgb

CLEANED_PATH = "HI-Small_cleaned.csv"
MODEL_PATH = "xgb_model.pkl"
TRUNCATE_DATE = "2022-09-11"
N_LEGIT_BACKGROUND = 50_000  # random legitimate transactions to sample alongside ALL laundering cases


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must exactly match the feature engineering in 03_features_and_model_v2.py."""
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


def describe_example(idx, X_sample, shap_values, y_true, y_proba, feature_names):
    row_shap = pd.Series(shap_values[idx], index=feature_names).sort_values(key=abs, ascending=False)
    return {
        "probability": round(float(y_proba[idx]), 4),
        "actual_is_laundering": int(y_true[idx]),
        "is_ach": int(X_sample.iloc[idx]['is_ach']),
        "amount_paid": round(float(X_sample.iloc[idx]['amount_paid']), 2),
        "top_5_contributing_features": {k: round(float(v), 4) for k, v in row_shap.head(5).items()},
    }


def main():
    print(f"Loading model from {MODEL_PATH} ...")
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)

    print(f"Loading {CLEANED_PATH} ...")
    df = pd.read_csv(CLEANED_PATH, parse_dates=['Timestamp'])
    df = df.sort_values('Timestamp').reset_index(drop=True)
    df = df[df['Timestamp'] < pd.Timestamp(TRUNCATE_DATE)].reset_index(drop=True)

    y = df['Is Laundering']
    X = engineer_features(df)

    n = len(df)
    val_end = int(n * 0.85)
    X_test = X.iloc[val_end:].reset_index(drop=True)
    y_test = y.iloc[val_end:].reset_index(drop=True)
    print(f"Test set: {len(X_test):,} rows, {int(y_test.sum())} laundering")

    # --- guaranteed-inclusion sample: ALL laundering cases + a random legit sample ---
    laundering_idx = y_test[y_test == 1].index.values
    legit_idx = y_test[y_test == 0].index.values
    rng = np.random.RandomState(42)
    legit_sample_idx = rng.choice(legit_idx, size=min(N_LEGIT_BACKGROUND, len(legit_idx)), replace=False)
    sample_idx = np.concatenate([laundering_idx, legit_sample_idx])
    rng.shuffle(sample_idx)

    X_sample = X_test.iloc[sample_idx].reset_index(drop=True)
    y_sample = y_test.iloc[sample_idx].reset_index(drop=True)
    print(f"SHAP sample: {len(X_sample):,} rows, {int(y_sample.sum())} laundering "
          f"(ALL {len(laundering_idx)} test laundering cases included, not randomly subsampled)")

    print("\nComputing SHAP values (TreeExplainer)...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    y_proba_sample = model.predict_proba(X_sample)[:, 1]
    y_pred_sample = (y_proba_sample >= 0.5).astype(int)

    # save for reuse without recomputation
    np.savez_compressed(
        'shap_values_full.npz',
        shap_values=shap_values,
        y_true=y_sample.values,
        y_proba=y_proba_sample,
        feature_names=np.array(X.columns.tolist(), dtype=object),
    )
    X_sample.to_csv('shap_sample_features.csv', index=False)
    print("Saved -> shap_values_full.npz, shap_sample_features.csv")

    # ---- global importance ----
    mean_abs_shap = pd.Series(np.abs(shap_values).mean(axis=0), index=X.columns).sort_values(ascending=False)
    print("\nTop 10 features by mean |SHAP value|:")
    print(mean_abs_shap.head(10).to_string())

    # ---- plots ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    shap.summary_plot(shap_values, X_sample, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig('shap_summary_plot.png', dpi=150, bbox_inches='tight')
    plt.close()

    shap.summary_plot(shap_values, X_sample, plot_type='bar', show=False, max_display=15)
    plt.tight_layout()
    plt.savefig('shap_summary_bar.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved -> shap_summary_plot.png, shap_summary_bar.png")

    # ---- THE KEY TEST: is amount_paid's influence suppressed for non-ACH transactions? ----
    amount_idx = X.columns.get_loc('amount_paid')
    ach_idx = X.columns.get_loc('is_ach')
    amount_shap = shap_values[:, amount_idx]
    ach_col = X_sample['is_ach'].values

    print("\n--- Testing: is amount_paid suppressed for non-ACH transactions? ---")
    result_by_group = {}
    for ach_val, label in [(1, "ACH transactions"), (0, "non-ACH transactions")]:
        mask = ach_col == ach_val
        if mask.sum() > 0:
            mean_amount_shap = float(amount_shap[mask].mean())
            mean_abs_amount_shap = float(np.abs(amount_shap[mask]).mean())
            print(f"{label} (n={mask.sum():,}): mean amount_paid SHAP = {mean_amount_shap:.4f}, "
                  f"mean |SHAP| = {mean_abs_amount_shap:.4f}")
            result_by_group[label] = {"n": int(mask.sum()), "mean_shap": round(mean_amount_shap, 4),
                                       "mean_abs_shap": round(mean_abs_amount_shap, 4)}

    # same test, restricted to actual laundering cases only -- the group that matters most
    print("\n--- Same test, laundering cases only ---")
    result_by_group_laundering = {}
    for ach_val, label in [(1, "ACH laundering"), (0, "non-ACH laundering")]:
        mask = (ach_col == ach_val) & (y_sample.values == 1)
        if mask.sum() > 0:
            mean_amount_shap = float(amount_shap[mask].mean())
            mean_prob = float(y_proba_sample[mask].mean())
            mean_amount = float(X_sample.loc[mask, 'amount_paid'].mean())
            print(f"{label} (n={mask.sum()}): mean amount_paid SHAP = {mean_amount_shap:.4f}, "
                  f"mean predicted probability = {mean_prob:.4f}, mean actual amount = ${mean_amount:,.0f}")
            result_by_group_laundering[label] = {
                "n": int(mask.sum()), "mean_amount_shap": round(mean_amount_shap, 4),
                "mean_predicted_probability": round(mean_prob, 4), "mean_actual_amount_usd": round(mean_amount, 2),
            }
        else:
            print(f"{label}: no cases in sample")
            result_by_group_laundering[label] = None

    # ---- stratified per-alert examples: TP, FP, FN ----
    print("\n--- Stratified example alerts ---")
    tp_mask = (y_pred_sample == 1) & (y_sample.values == 1)
    fp_mask = (y_pred_sample == 1) & (y_sample.values == 0)
    fn_mask = (y_pred_sample == 0) & (y_sample.values == 1)

    examples = {}
    for name, mask in [("true_positive", tp_mask), ("false_positive", fp_mask), ("false_negative", fn_mask)]:
        idxs = np.where(mask)[0][:2]  # up to 2 examples each
        examples[name] = [
            describe_example(i, X_sample, shap_values, y_sample.values, y_proba_sample, X.columns.tolist())
            for i in idxs
        ]
        print(f"\n{name} ({mask.sum()} available in sample, showing up to 2):")
        for ex in examples[name]:
            print(f"  probability={ex['probability']}, is_ach={ex['is_ach']}, amount=${ex['amount_paid']:,.0f}")
            for feat_name, val in ex['top_5_contributing_features'].items():
                print(f"      {feat_name}: {val:+.4f}")

    summary = {
        "n_sample": len(X_sample),
        "n_laundering_in_sample": int(y_sample.sum()),
        "note": "ALL test-set laundering cases included (not randomly subsampled)",
        "top_10_global_shap_importance": mean_abs_shap.head(10).to_dict(),
        "amount_paid_suppression_test_all_transactions": result_by_group,
        "amount_paid_suppression_test_laundering_only": result_by_group_laundering,
        "stratified_examples": examples,
        "output_files": ["shap_summary_plot.png", "shap_summary_bar.png",
                          "shap_values_full.npz", "shap_sample_features.csv"],
    }

    print("\n" + "=" * 60)
    print("===SUMMARY_START===")
    print(json.dumps(summary, indent=2, default=str))
    print("===SUMMARY_END===")
    print("=" * 60)
    print("\nCopy everything between the markers and send it back, plus the two PNG files if you can.")


if __name__ == "__main__":
    main()
