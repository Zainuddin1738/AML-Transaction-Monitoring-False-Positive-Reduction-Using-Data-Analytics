"""
Quick sanity check BEFORE running 03_features_and_model.py.

Checks:
0. ACCOUNT ID INTEGRITY -- were From Account / To Account corrupted into
   scientific notation (e.g. "8.00E+06") by being opened in Excel at some
   point? This silently collapses distinct accounts into one and would
   break every account-level feature downstream. Checked first because
   nothing else in this script (or the modelling script) is trustworthy
   if this fails.
1. Are laundering cases spread reasonably evenly across the 17-day window,
   or clustered in specific days/hours? (validates the split is fair)
2. Are laundering cases concentrated in a small number of accounts, or
   spread across many? (affects how much the model can generalize, and
   whether train/test see overlapping vs. distinct laundering behaviour)
3. Does the "Reinvestment" payment format = self-transaction = never
   laundering pattern hold across the FULL dataset (only eyeballed on a
   small sample so far)? If confirmed, these can be safely excluded from
   the alerting universe, improving precision for free.
4. Currency breakdown -- the sample shown was 100% US Dollar; confirm
   whether other currencies exist and in what volume.

This is deliberately lightweight -- pure aggregation, runs in seconds even
on the full 5M-row file.

Usage:
    python 01b_temporal_and_account_check.py
"""

import json
import re
import pandas as pd
import numpy as np

CLEANED_PATH = "HI-Small_cleaned.csv"

# Expected rough account count for HI-Small per the IBM AMLworld paper (~515K).
# A count far below this is a red flag for ID collapse/corruption.
EXPECTED_APPROX_ACCOUNTS = 515_000


def check_account_id_integrity(df: pd.DataFrame) -> dict:
    scientific_notation_pattern = re.compile(r'^\d(\.\d+)?E[+-]\d+$', re.IGNORECASE)

    from_dtype = str(df['From Account'].dtype)
    to_dtype = str(df['To Account'].dtype)

    from_sample = df['From Account'].astype(str)
    to_sample = df['To Account'].astype(str)

    n_from_scientific = int(from_sample.str.match(scientific_notation_pattern).sum())
    n_to_scientific = int(to_sample.str.match(scientific_notation_pattern).sum())

    n_unique_from = df['From Account'].nunique()
    n_unique_to = df['To Account'].nunique()
    n_unique_total = pd.concat([df['From Account'], df['To Account']]).nunique()

    return {
        "from_account_dtype": from_dtype,
        "to_account_dtype": to_dtype,
        "n_rows_from_account_scientific_notation": n_from_scientific,
        "n_rows_to_account_scientific_notation": n_to_scientific,
        "n_unique_from_accounts": int(n_unique_from),
        "n_unique_to_accounts": int(n_unique_to),
        "n_unique_accounts_total": int(n_unique_total),
        "expected_approx_accounts_per_paper": EXPECTED_APPROX_ACCOUNTS,
        "unique_account_count_looks_reasonable": bool(n_unique_total > EXPECTED_APPROX_ACCOUNTS * 0.5),
    }


def main():
    print(f"Loading {CLEANED_PATH} ...")
    df = pd.read_csv(CLEANED_PATH, parse_dates=['Timestamp'], dtype={'From Account': str, 'To Account': str})
    df = df.sort_values('Timestamp').reset_index(drop=True)
    print(f"Loaded shape: {df.shape}")

    print("Checking account ID integrity...")
    account_integrity = check_account_id_integrity(df)
    if account_integrity["n_rows_from_account_scientific_notation"] > 0 or \
       account_integrity["n_rows_to_account_scientific_notation"] > 0:
        print("\n*** WARNING: scientific-notation account IDs detected. ***")
        print("*** This CSV was likely corrupted by being opened/saved in Excel. ***")
        print("*** Re-export from the original Kaggle download before proceeding. ***\n")
    if not account_integrity["unique_account_count_looks_reasonable"]:
        print(f"\n*** WARNING: only {account_integrity['n_unique_accounts_total']:,} unique accounts found,")
        print(f"*** well below the ~{EXPECTED_APPROX_ACCOUNTS:,} expected for HI-Small.")
        print("*** This suggests account IDs have collapsed/lost precision somewhere in the pipeline.\n")

    # --- 1. daily laundering distribution across the full date range ---
    df['date'] = df['Timestamp'].dt.date
    daily = df.groupby('date').agg(
        n_transactions=('Is Laundering', 'size'),
        n_laundering=('Is Laundering', 'sum')
    )
    daily['laundering_rate_pct'] = (daily['n_laundering'] / daily['n_transactions'] * 100).round(4)
    daily_dict = {str(k): v for k, v in daily.to_dict('index').items()}

    # --- 2. where would the 70/15/15 temporal split actually land? ---
    n = len(df)
    train_end, val_end = int(n * 0.70), int(n * 0.85)
    split_dates = {
        "train_end_date": str(df['Timestamp'].iloc[train_end]),
        "val_end_date": str(df['Timestamp'].iloc[val_end - 1]),
        "test_end_date": str(df['Timestamp'].iloc[-1]),
    }
    n_laundering_train = int(df['Is Laundering'].iloc[:train_end].sum())
    n_laundering_val = int(df['Is Laundering'].iloc[train_end:val_end].sum())
    n_laundering_test = int(df['Is Laundering'].iloc[val_end:].sum())

    # --- 3. account concentration ---
    laundering_df = df[df['Is Laundering'] == 1]
    from_accounts = laundering_df['From Account'].nunique()
    to_accounts = laundering_df['To Account'].nunique()
    all_from_accounts = df['From Account'].nunique()

    top_senders = laundering_df['From Account'].value_counts().head(10).to_dict()
    top_senders = {str(k): int(v) for k, v in top_senders.items()}

    # do accounts involved in laundering appear in BOTH the would-be train and test windows?
    train_laundering_accounts = set(laundering_df[laundering_df['Timestamp'] < df['Timestamp'].iloc[train_end]]['From Account'])
    test_laundering_accounts = set(laundering_df[laundering_df['Timestamp'] >= df['Timestamp'].iloc[val_end]]['From Account'])
    overlap = train_laundering_accounts & test_laundering_accounts

    # --- reinvestment / self-transaction structural check ---
    print("Checking Reinvestment / self-transaction pattern...")
    is_self = (df['From Account'] == df['To Account']) & (df['From Bank'] == df['To Bank'])
    is_reinvestment = df['Payment Format'] == 'Reinvestment'

    n_reinvestment = int(is_reinvestment.sum())
    n_reinvestment_and_self = int((is_reinvestment & is_self).sum())
    pct_reinvestment_is_self = round(n_reinvestment_and_self / n_reinvestment * 100, 2) if n_reinvestment else None
    reinvestment_laundering_count = int(df.loc[is_reinvestment, 'Is Laundering'].sum())

    n_self = int(is_self.sum())
    n_self_and_reinvestment = n_reinvestment_and_self
    pct_self_is_reinvestment = round(n_self_and_reinvestment / n_self * 100, 2) if n_self else None
    self_laundering_count = int(df.loc[is_self, 'Is Laundering'].sum())

    # --- currency breakdown ---
    payment_currency_counts = df['Payment Currency'].value_counts().to_dict()
    receiving_currency_counts = df['Receiving Currency'].value_counts().to_dict()

    summary = {
        "account_id_integrity": account_integrity,
        "reinvestment_self_transaction_check": {
            "n_reinvestment_transactions": n_reinvestment,
            "pct_of_reinvestment_that_is_self_transaction": pct_reinvestment_is_self,
            "reinvestment_laundering_count": reinvestment_laundering_count,
            "n_self_transactions": n_self,
            "pct_of_self_transactions_that_is_reinvestment_format": pct_self_is_reinvestment,
            "self_transaction_laundering_count": self_laundering_count,
        },
        "currency_breakdown": {
            "payment_currency_counts": payment_currency_counts,
            "receiving_currency_counts": receiving_currency_counts,
        },
        "date_range": [str(df['Timestamp'].min()), str(df['Timestamp'].max())],
        "n_days": len(daily),
        "daily_laundering_counts": {k: int(v['n_laundering']) for k, v in daily_dict.items()},
        "daily_laundering_rate_pct": {k: v['laundering_rate_pct'] for k, v in daily_dict.items()},
        "proposed_split_dates": split_dates,
        "laundering_count_by_split": {
            "train": n_laundering_train,
            "val": n_laundering_val,
            "test": n_laundering_test,
        },
        "unique_laundering_from_accounts": from_accounts,
        "unique_laundering_to_accounts": to_accounts,
        "total_unique_from_accounts": all_from_accounts,
        "pct_of_accounts_involved_in_laundering": round(from_accounts / all_from_accounts * 100, 4),
        "top_10_laundering_sender_accounts": top_senders,
        "laundering_accounts_appearing_in_both_train_and_test_windows": len(overlap),
    }

    print("\n" + "=" * 60)
    print("===SUMMARY_START===")
    print(json.dumps(summary, indent=2, default=str))
    print("===SUMMARY_END===")
    print("=" * 60)
    print("\nCopy everything between the SUMMARY_START / SUMMARY_END markers and send it back.")


if __name__ == "__main__":
    main()
