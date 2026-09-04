"""
v2 -- corrects the threshold-calibration problem found on the full dataset.

Changes from v1:
1. high_value / structuring: reverted to the REAL regulatory CTR threshold
   ($10,000), not a data-fit percentile. These rules exist to mirror actual
   law -- percentile-fitting them doesn't make sense. (v1's 99th-percentile
   approach got dragged to $13.5M by extreme outlier transactions and became
   useless -- a good lesson for the methodology section.)
2. high_velocity: fixed, domain-reasoned threshold (10 txns/24h) instead of
   a percentile that was dominated by a handful of outlier "hub" accounts.
3. high_risk_format: now targets ACH (empirically ~7.3x baseline laundering
   rate in this dataset) instead of the conventional Bitcoin/Cash assumption,
   which was found to be BELOW the baseline rate here. Bitcoin/Cash kept
   available as a togglable option for regulatory completeness/discussion.

Run locally after 01_clean_data_full.py, on the full cleaned dataset.
Usage:
    python 02_rule_baseline_full_v2.py
"""

import json
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score

# ---- config ----
CLEANED_PATH = "HI-Small_cleaned.csv"
OUT_PATH = "rule_based_baseline_predictions_v2.csv"

HIGH_VALUE_THRESHOLD = 10_000          # real CTR reporting threshold, not data-fit
STRUCTURING_BAND = (9_000, 10_000)     # just-under-threshold evasion band
VELOCITY_WINDOW = "24h"
VELOCITY_MIN_TXNS = 10                 # fixed, domain-reasoned (tune if needed)
ROUND_AMOUNT_MULTIPLE = 1_000
HIGH_RISK_FORMATS = ("ACH",)           # empirically-informed for this dataset

# Same stable-window restriction as 03_features_and_model.py -- see that script's
# comment for why: daily laundering rate explodes to 57-73% after Sep 10 due to a
# collapse in background transaction volume (simulator artifact, not a real signal).
TRUNCATE_DATE = "2022-09-11"  # exclusive -- keep Sep 1 through Sep 10 only
# HIGH_RISK_FORMATS = ("ACH", "Bitcoin", "Cash")  # alt: regulatory-conventional + empirical


def main():
    print(f"Loading {CLEANED_PATH} ...")
    df = pd.read_csv(CLEANED_PATH, parse_dates=['Timestamp'])
    print(f"Loaded shape: {df.shape}")

    n_before = len(df)
    df = df[df['Timestamp'] < pd.Timestamp(TRUNCATE_DATE)].reset_index(drop=True)
    print(f"Truncated to {TRUNCATE_DATE} (excl.): {n_before:,} -> {len(df):,} rows")
    print(f"Laundering cases: {int(df['Is Laundering'].sum())} out of {len(df):,}")

    # --- apply rules ---
    flags = pd.DataFrame(index=df.index)
    flags['high_value'] = df['Amount Paid'] > HIGH_VALUE_THRESHOLD
    flags['structuring'] = (df['Amount Paid'] >= STRUCTURING_BAND[0]) & (df['Amount Paid'] < STRUCTURING_BAND[1])

    print("Computing account velocity (this can take a minute on the full file)...")
    tmp = df[['From Account', 'Timestamp']].copy()
    tmp['ones'] = 1
    tmp = tmp.sort_values(['From Account', 'Timestamp'])
    rolling_counts = (
        tmp.set_index('Timestamp')
           .groupby('From Account')['ones']
           .rolling(VELOCITY_WINDOW)
           .sum()
           .reset_index(drop=True)
    )
    tmp = tmp.reset_index(drop=True)
    tmp['rolling_count'] = rolling_counts.values
    velocity_flag_by_pos = tmp['rolling_count'] >= VELOCITY_MIN_TXNS
    velocity_result = pd.Series(False, index=df.index)
    velocity_result.iloc[tmp.index.values] = velocity_flag_by_pos.values
    flags['high_velocity'] = velocity_result

    flags['round_amount'] = (df['Amount Paid'] % ROUND_AMOUNT_MULTIPLE == 0) & (df['Amount Paid'] > 0)
    flags['high_risk_format'] = df['Payment Format'].isin(HIGH_RISK_FORMATS)

    flags['alert'] = flags[['high_value', 'structuring', 'high_velocity', 'round_amount', 'high_risk_format']].any(axis=1)

    # --- rule firing frequency ---
    rule_counts = {k: int(v) for k, v in flags.drop(columns='alert').sum().to_dict().items()}

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

    # --- payment format breakdown (for reference / sanity check against cleaning summary) ---
    fmt_laundering_rate = (df.groupby('Payment Format')['Is Laundering'].mean() * 100).round(4).to_dict()

    # --- save predictions ---
    out = df[['Timestamp', 'From Account', 'To Account', 'Amount Paid', 'Payment Format', 'Is Laundering']].copy()
    out['rule_based_alert'] = y_pred.values
    out.to_csv(OUT_PATH, index=False)
    print(f"\nSaved predictions -> {OUT_PATH}")

    summary = {
        "n_rows": int(len(df)),
        "n_laundering": int(df['Is Laundering'].sum()),
        "thresholds_used": {
            "high_value_threshold": HIGH_VALUE_THRESHOLD,
            "structuring_band": list(STRUCTURING_BAND),
            "velocity_min_txns_per_24h": VELOCITY_MIN_TXNS,
            "high_risk_formats": list(HIGH_RISK_FORMATS),
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
        "payment_format_laundering_rate_pct": fmt_laundering_rate,
    }

    print("\n" + "=" * 60)
    print("===SUMMARY_START===")
    print(json.dumps(summary, indent=2, default=str))
    print("===SUMMARY_END===")
    print("=" * 60)
    print("\nCopy everything between the SUMMARY_START / SUMMARY_END markers and send it back.")


if __name__ == "__main__":
    main()
