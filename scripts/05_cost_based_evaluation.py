"""
Formal cost-based evaluation -- Objective 6 explicitly names "cost-based metrics",
which we hadn't actually computed yet (alerts-per-catch is a proxy, not a real
cost model). This script fixes that.

Uses files you already have -- NO retraining required:
  - rule_based_baseline_predictions_v2.csv  (from 02_rule_baseline_full_v2.py)
  - ml_predictions.csv                       (from 03_features_and_model.py or v2)

Cost model (stated explicitly so it can be justified/cited or adjusted in your
write-up -- these are illustrative assumptions, not a definitive industry figure):
  - False positive cost: a flat per-alert review cost (analyst time). Default
    $30/alert -- adjust COST_PER_ALERT_REVIEW below, ideally citing a source
    from your literature review (Anaptyss 2025 / Flagright 2026 may have a
    more precise, citable figure).
  - False negative cost: the ACTUAL Amount Paid of the missed transaction --
    a data-grounded proxy for direct financial exposure from undetected
    laundering (not a guess; it's the literal amount that got through). Real
    institutional costs would also include regulatory fines and reputational
    risk beyond this, worth naming as a limitation.

Compares:
  1. Rule-based baseline cost (restricted to the SAME time window as the ML
     test set, for a fair comparison -- the baseline file covers the whole
     dataset, the ML file only covers its test split)
  2. ML at default threshold (0.5)
  3. ML swept across all thresholds -- finds the COST-MINIMIZING threshold,
     which is the "threshold calibration" the literature review specifically
     names as effective for imbalanced classification

Usage:
    python 05_cost_based_evaluation.py
"""

import json
import pandas as pd
import numpy as np

RULE_PREDICTIONS_PATH = "rule_based_baseline_predictions_v2.csv"
ML_PREDICTIONS_PATH = "ml_predictions.csv"

COST_PER_ALERT_REVIEW = 30.0  # USD, illustrative -- adjust/cite as appropriate


def compute_cost(n_false_positives, false_negative_amounts, cost_per_alert):
    fp_cost = n_false_positives * cost_per_alert
    fn_cost = float(np.sum(false_negative_amounts))
    return {
        "fp_cost": round(fp_cost, 2),
        "fn_cost": round(fn_cost, 2),
        "total_cost": round(fp_cost + fn_cost, 2),
        "n_false_positives": int(n_false_positives),
        "n_false_negatives": int(len(false_negative_amounts)),
    }


def main():
    print(f"Loading {ML_PREDICTIONS_PATH} ...")
    ml_df = pd.read_csv(ML_PREDICTIONS_PATH, parse_dates=['Timestamp'])
    test_start, test_end = ml_df['Timestamp'].min(), ml_df['Timestamp'].max()
    print(f"ML test window: {test_start} to {test_end} ({len(ml_df):,} rows)")

    print(f"Loading {RULE_PREDICTIONS_PATH} ...")
    rule_df = pd.read_csv(RULE_PREDICTIONS_PATH, parse_dates=['Timestamp'])

    # restrict rule baseline to the SAME time window as the ML test set, for a fair comparison
    rule_test_window = rule_df[
        (rule_df['Timestamp'] >= test_start) & (rule_df['Timestamp'] <= test_end)
    ].copy()
    print(f"Rule baseline restricted to matching window: {len(rule_test_window):,} rows "
          f"(full baseline file was {len(rule_df):,} rows)")

    if len(rule_test_window) != len(ml_df):
        print("WARNING: row counts don't match exactly after windowing -- comparing on "
              "overlapping time range, minor discrepancies possible from split-boundary rounding.")

    # ---- 1. rule-based baseline cost, on the matching window ----
    rule_fp_mask = (rule_test_window['rule_based_alert'] == 1) & (rule_test_window['Is Laundering'] == 0)
    rule_fn_mask = (rule_test_window['rule_based_alert'] == 0) & (rule_test_window['Is Laundering'] == 1)
    rule_cost = compute_cost(
        rule_fp_mask.sum(),
        rule_test_window.loc[rule_fn_mask, 'Amount Paid'].values,
        COST_PER_ALERT_REVIEW,
    )
    print(f"\nRule-based baseline cost (matching window): {rule_cost}")

    # ---- 2. ML at default threshold ----
    ml_fp_mask = (ml_df['ml_alert_default_threshold'] == 1) & (ml_df['Is Laundering'] == 0)
    ml_fn_mask = (ml_df['ml_alert_default_threshold'] == 0) & (ml_df['Is Laundering'] == 1)
    ml_default_cost = compute_cost(
        ml_fp_mask.sum(),
        ml_df.loc[ml_fn_mask, 'Amount Paid'].values,
        COST_PER_ALERT_REVIEW,
    )
    print(f"ML (default threshold 0.5) cost: {ml_default_cost}")

    # ---- 3. sweep ML thresholds to find the cost-minimizing one ----
    print("\nSweeping thresholds to find cost-minimizing operating point...")
    thresholds_to_try = np.linspace(0.01, 0.99, 99)
    sweep_results = []
    for t in thresholds_to_try:
        pred = (ml_df['ml_probability'] >= t).astype(int)
        fp_mask = (pred == 1) & (ml_df['Is Laundering'] == 0)
        fn_mask = (pred == 0) & (ml_df['Is Laundering'] == 1)
        c = compute_cost(fp_mask.sum(), ml_df.loc[fn_mask, 'Amount Paid'].values, COST_PER_ALERT_REVIEW)
        c['threshold'] = round(float(t), 3)
        sweep_results.append(c)

    sweep_df = pd.DataFrame(sweep_results)
    best_row = sweep_df.loc[sweep_df['total_cost'].idxmin()]
    print(f"\nCost-minimizing threshold: {best_row['threshold']}")
    print(f"  Total cost: ${best_row['total_cost']:,.2f}")
    print(f"  FP cost: ${best_row['fp_cost']:,.2f} ({int(best_row['n_false_positives'])} false positives)")
    print(f"  FN cost: ${best_row['fn_cost']:,.2f} ({int(best_row['n_false_negatives'])} false negatives)")

    sweep_df.to_csv("cost_sweep_data.csv", index=False)
    print("\nSaved full threshold sweep -> cost_sweep_data.csv (for plotting cost vs. threshold)")

    # ---- comparison summary ----
    cost_reduction_default = rule_cost['total_cost'] / ml_default_cost['total_cost'] if ml_default_cost['total_cost'] > 0 else None
    cost_reduction_optimal = rule_cost['total_cost'] / best_row['total_cost'] if best_row['total_cost'] > 0 else None

    summary = {
        "cost_assumptions": {
            "cost_per_alert_review_usd": COST_PER_ALERT_REVIEW,
            "false_negative_cost_basis": "actual Amount Paid of the missed laundering transaction",
        },
        "comparison_window": {"start": str(test_start), "end": str(test_end), "n_transactions": len(ml_df)},
        "rule_based_baseline_cost": rule_cost,
        "ml_default_threshold_cost": ml_default_cost,
        "ml_cost_minimizing_threshold": {
            "threshold": float(best_row['threshold']),
            "total_cost": float(best_row['total_cost']),
            "fp_cost": float(best_row['fp_cost']),
            "fn_cost": float(best_row['fn_cost']),
            "n_false_positives": int(best_row['n_false_positives']),
            "n_false_negatives": int(best_row['n_false_negatives']),
        },
        "cost_reduction_vs_baseline": {
            "at_default_threshold": round(cost_reduction_default, 2) if cost_reduction_default else None,
            "at_cost_minimizing_threshold": round(cost_reduction_optimal, 2) if cost_reduction_optimal else None,
        },
    }

    print("\n" + "=" * 60)
    print("===SUMMARY_START===")
    print(json.dumps(summary, indent=2, default=str))
    print("===SUMMARY_END===")
    print("=" * 60)
    print("\nCopy everything between the SUMMARY_START / SUMMARY_END markers and send it back.")
    print("Also send cost_sweep_data.csv if you can.")


if __name__ == "__main__":
    main()
