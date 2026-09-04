"""
Run locally AFTER 01_clean_data_full.py, on the full cleaned dataset.

Applies a rule-based AML alert baseline with THRESHOLDS CALIBRATED TO YOUR
DATA (percentile-based) rather than a fixed $10,000 figure -- this fixes the
miscalibration issue found on the small dev sample, where a fixed $10k
threshold flagged ~38% of all transactions (unrealistic; real systems flag
a low single-digit percentage).

Prints a SUMMARY block -- copy it back the same way as script 01.

Usage:
    python 02_rule_baseline_full.py
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ---- config ----
CLEANED_PATH = "HI-Small_cleaned.csv"
OUT_PATH = "rule_based_baseline_predictions.csv"

HIGH_VALUE_PERCENTILE = 99        # flag top 1% of transaction amounts
STRUCTURING_BAND_PCT = 0.10       # "just under" = within 10% below the high-value threshold
VELOCITY_WINDOW = "24h"
VELOCITY_PERCENTILE = 99          # flag accounts whose 24h txn count is in the top 1%
ROUND_AMOUNT_MULTIPLE = 1000
HIGH_RISK_FORMATS = ("Bitcoin", "Cash")


def flag_high_velocity(df: pd.DataFrame, window: str, min_txns: int) -> pd.Series:
    tmp = df[['From Account', 'Timestamp']].copy()
    tmp['ones'] = 1
    tmp = tmp.sort_values(['From Account', 'Timestamp'])
    counts = (
        tmp.set_index('Timestamp')
           .groupby('From Account')['ones']
           .rolling(window)
           .sum()
           .reset_index(drop=True)
    )
    tmp = tmp.reset_index(drop=True)
    tmp['count'] = counts
    flags = tmp['count'] >= min_txns
    flags.index = df.index[tmp.index] if len(tmp) == len(df) else df.index
    # realign safely by sorting back to original order
    result = pd.Series(False, index=df.index)
    result.loc[tmp.index] = flags.values
    return result


def main():
    print(f"Loading {CLEANED_PATH} ...")
    df = pd.read_csv(CLEANED_PATH, parse_dates=['Timestamp'])
    print(f"Loaded shape: {df.shape}")
    print(f"Laundering cases: {int(df['Is Laundering'].sum())} out of {len(df):,}")

    # --- calibrate thresholds from the data itself ---
    high_value_threshold = float(np.percentile(df['Amount Paid'], HIGH_VALUE_PERCENTILE))
    structuring_lower = high_value_threshold * (1 - STRUCTURING_BAND_PCT)

    print(f"\nCalibrated high-value threshold (p{HIGH_VALUE_PERCENTILE}): {high_value_threshold:,.2f}")
    print(f"Structuring band: [{structuring_lower:,.2f}, {high_value_threshold:,.2f})")

    # --- apply rules ---
    flags = pd.DataFrame(index=df.index)
    flags['high_value'] = df['Amount Paid'] > high_value_threshold
    flags['structuring'] = (df['Amount Paid'] >= structuring_lower) & (df['Amount Paid'] < high_value_threshold)

    # velocity threshold: compute a rough per-account 24h count distribution to calibrate min_txns
    print("Computing account velocity (this can take a minute on the full file)...")
    tmp = df[['From Account', 'Timestamp']].copy()
    tmp['ones'] = 1
    tmp = tmp.sort_values(['From Account', 'Timestamp'])
    rolling_counts = (
        tmp.set_index('Timestamp')
           .groupby('From Account')['ones']
           .rolling(VELOCITY_WINDOW)
           .sum()
    )
    rolling_counts = rolling_counts.reset_index(drop=True)
    velocity_threshold = float(np.percentile(rolling_counts, VELOCITY_PERCENTILE))
    velocity_threshold = max(velocity_threshold, 3)  # floor at 3 so it's not trivially 1-2
    print(f"Calibrated velocity threshold (p{VELOCITY_PERCENTILE} of 24h txn count): {velocity_threshold:.1f}")

    tmp = tmp.reset_index(drop=True)
    tmp['rolling_count'] = rolling_counts.values
    velocity_flag_by_pos = tmp['rolling_count'] >= velocity_threshold
    velocity_result = pd.Series(False, index=df.index)
    velocity_result.iloc[tmp.index.values] = velocity_flag_by_pos.values
    flags['high_velocity'] = velocity_result

    flags['round_amount'] = (df['Amount Paid'] % ROUND_AMOUNT_MULTIPLE == 0) & (df['Amount Paid'] > 0)
    flags['high_risk_format'] = df['Payment Format'].isin(HIGH_RISK_FORMATS)

    flags['alert'] = flags[['high_value', 'structuring', 'high_velocity', 'round_amount', 'high_risk_format']].any(axis=1)

    # --- rule firing frequency ---
    rule_counts = flags.drop(columns='alert').sum().to_dict()
    rule_counts = {k: int(v) for k, v in rule_counts.items()}

    # --- overall alert volume ---
    n_alerts = int(flags['alert'].sum())
    alert_rate_pct = round(n_alerts / len(df) * 100, 3)

    # --- evaluation ---
    y_true = df['Is Laundering']
    y_pred = flags['alert'].astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    alerts_per_catch = round((tp + fp) / tp, 1) if tp > 0 else None

    # --- per-rule diagnostic ---
    diagnostic = flags.copy()
    diagnostic['Is Laundering'] = df['Is Laundering'].values
    per_rule = {}
    for rule_name in ['high_value', 'structuring', 'high_velocity', 'round_amount', 'high_risk_format']:
        fired = diagnostic[diagnostic[rule_name]]
        per_rule[rule_name] = {
            "n_fired": int(len(fired)),
            "n_laundering_caught": int(fired['Is Laundering'].sum()) if len(fired) else 0,
            "hit_rate_pct": round(fired['Is Laundering'].mean() * 100, 4) if len(fired) else 0.0,
        }

    # --- save predictions ---
    out = df[['Timestamp', 'From Account', 'To Account', 'Amount Paid', 'Is Laundering']].copy()
    out['rule_based_alert'] = y_pred.values
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved predictions -> {OUT_PATH}")

    summary = {
        "n_rows": int(len(df)),
        "n_laundering": int(df['Is Laundering'].sum()),
        "calibrated_thresholds": {
            "high_value_threshold": round(high_value_threshold, 2),
            "structuring_band": [round(structuring_lower, 2), round(high_value_threshold, 2)],
            "velocity_threshold_txns_per_24h": round(velocity_threshold, 1),
        },
        "rule_fire_counts": rule_counts,
        "n_alerts": n_alerts,
        "alert_rate_pct": alert_rate_pct,
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "alerts_per_true_catch": alerts_per_catch,
        "per_rule_breakdown": per_rule,
    }

    print("\n" + "=" * 60)
    print("===SUMMARY_START===")
    print(json.dumps(summary, indent=2, default=str))
    print("===SUMMARY_END===")
    print("=" * 60)
    print("\nCopy everything between the SUMMARY_START / SUMMARY_END markers and send it back.")


if __name__ == "__main__":
    main()
