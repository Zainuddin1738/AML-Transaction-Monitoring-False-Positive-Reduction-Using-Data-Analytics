"""
Run locally on the FULL HI-Small_Trans.csv.

Cleans the data and prints a SUMMARY block at the end -- copy that whole
block (between the markers) and send it back so the notebook narrative and
next script can be built using your real numbers instead of a small sample.

Usage:
    python 01_clean_data_full.py
"""

import json
import pandas as pd
import numpy as np

# ---- config ----
RAW_PATH = "HI-Small_Trans.csv"          # adjust if your file is elsewhere
OUT_PATH = "HI-Small_cleaned.csv"        # cleaned output, kept local


def main():
    print(f"Loading {RAW_PATH} ...")
    df = pd.read_csv(RAW_PATH)
    print(f"Loaded shape: {df.shape}")

    # --- schema check ---
    expected_cols = {
        'Timestamp', 'From Bank', 'Account', 'To Bank', 'Account.1',
        'Amount Received', 'Receiving Currency', 'Amount Paid',
        'Payment Currency', 'Payment Format', 'Is Laundering'
    }
    missing_cols = expected_cols - set(df.columns)
    if missing_cols:
        print(f"WARNING: expected columns not found: {missing_cols}")

    # --- missing values / duplicates ---
    missing = df.isnull().sum()
    n_missing_total = int(missing.sum())
    n_dupes = int(df.duplicated().sum())

    # --- timestamp parsing ---
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    date_min, date_max = df['Timestamp'].min(), df['Timestamp'].max()
    span_days = (date_max - date_min).days

    # --- rename ambiguous columns ---
    df = df.rename(columns={'Account': 'From Account', 'Account.1': 'To Account'})

    # --- class balance ---
    class_counts = df['Is Laundering'].value_counts().to_dict()
    n_laundering = int(class_counts.get(1, 0))
    n_legit = int(class_counts.get(0, 0))
    laundering_rate_1_in_n = round(len(df) / n_laundering, 1) if n_laundering else None

    # --- currency mismatch feature ---
    df['currency_mismatch'] = (df['Payment Currency'] != df['Receiving Currency']).astype(int)
    mismatch_laundering_rate = df.groupby('currency_mismatch')['Is Laundering'].mean().to_dict()

    # --- payment format breakdown ---
    fmt_counts = df['Payment Format'].value_counts().to_dict()
    fmt_laundering_rate = (df.groupby('Payment Format')['Is Laundering'].mean() * 100).round(4).to_dict()

    # --- amount stats by class ---
    amount_stats = df.groupby('Is Laundering')['Amount Paid'].describe().round(2).to_dict()

    # --- time patterns ---
    df['hour'] = df['Timestamp'].dt.hour
    df['day_of_week'] = df['Timestamp'].dt.day_name()
    hourly_rate = (df.groupby('hour')['Is Laundering'].mean() * 100).round(4).to_dict()

    # --- sanity checks ---
    n_nonpositive_paid = int((df['Amount Paid'] <= 0).sum())
    n_nonpositive_received = int((df['Amount Received'] <= 0).sum())
    self_txn = int(((df['From Account'] == df['To Account']) & (df['From Bank'] == df['To Bank'])).sum())

    # --- save cleaned data ---
    df.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned data -> {OUT_PATH}")

    summary = {
        "shape": list(df.shape),
        "columns": df.columns.tolist(),
        "n_missing_total": n_missing_total,
        "n_duplicate_rows": n_dupes,
        "date_range": [str(date_min), str(date_max)],
        "span_days": span_days,
        "class_counts": class_counts,
        "laundering_rate_1_in_n": laundering_rate_1_in_n,
        "currency_mismatch_laundering_rate": mismatch_laundering_rate,
        "payment_format_counts": fmt_counts,
        "payment_format_laundering_rate_pct": fmt_laundering_rate,
        "amount_paid_stats_by_class": amount_stats,
        "hourly_laundering_rate_pct": hourly_rate,
        "n_nonpositive_amount_paid": n_nonpositive_paid,
        "n_nonpositive_amount_received": n_nonpositive_received,
        "n_self_transactions": self_txn,
    }

    print("\n" + "=" * 60)
    print("===SUMMARY_START===")
    print(json.dumps(summary, indent=2, default=str))
    print("===SUMMARY_END===")
    print("=" * 60)
    print("\nCopy everything between the SUMMARY_START / SUMMARY_END markers and send it back.")


if __name__ == "__main__":
    main()
