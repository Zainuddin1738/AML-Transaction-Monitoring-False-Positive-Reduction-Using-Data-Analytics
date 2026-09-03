# AML Transaction Monitoring — False-Positive Reduction Using Data Analytics

Cost-sensitive ML pipeline for AML alert triage vs. a rule-based baseline, with SHAP
explainability. MSc Big Data with Banking and Finance — Research Skills for Computing (55-710248).

## Research question

To what extent can an ML-based monitoring pipeline reduce false-positive AML alerts vs. a
rule-based baseline, without compromising detection of genuine activity?

## Setup

```bash
pip install -r requirements.txt
```

## Getting the data

1. Download `HI-Small_Trans.csv` from the [Kaggle dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml)
2. Place it in `data/raw/HI-Small_Trans.csv`
3. Run the scripts in `local_scripts/` in order (see below) — each prints a `SUMMARY` block used to
   build the corresponding notebook

## Project structure

```
aml-project/
├── data/
│   ├── raw/            # HI-Small_Trans.csv goes here (not tracked in git)
│   └── processed/       # cleaned data, model artifacts (not tracked in git)
├── notebooks/           # five notebooks, executed with real results
├── local_scripts/        # standalone scripts, run locally against the full dataset
├── src/
│   └── rules.py          # reusable rule-based AML flagging module
├── requirements.txt
└── README.md
```

## Pipeline (run in this order)

| Step | Script | Notebook | Objective |
|---|---|---|---|
| 1 | `01_clean_data_full.py` | `01_data_cleaning.ipynb` | Obj 2 — dataset prep, variant justification |
| 1b | `01b_temporal_and_account_check.py` | (feeds into 01) | — data integrity & temporal validity checks |
| 2 | `02_rule_baseline_full_v2.py` | `02_rule_based_baseline.ipynb` | Obj 3 — rule-based baseline |
| 3 | `03_features_and_model_v2.py` | `03_modelling.ipynb` | Obj 4 — cost-sensitive classifier(s) |
| 4 | `05_cost_based_evaluation.py` | `04_evaluation.ipynb` | Obj 6 — PR-AUC, F1, cost-based comparison |
| 5 | `04_shap_explainability_v2.py` | `05_shap_explainability.ipynb` | Obj 5 — SHAP explainability |

`v1` script versions (`02_rule_baseline_full.py`, `03_features_and_model.py`,
`04_shap_explainability.py`) are kept in the repo deliberately — they represent real earlier
attempts (percentile-calibrated thresholds, missing early stopping, an under-sampled SHAP run) that
didn't work as well as the final versions. The corresponding notebooks document what went wrong and
why, rather than only showing the final polished result.

## Key findings

- **HI-Small variant**, justified against the other 5 IBM AMLworld variants (`01_data_cleaning.ipynb`, Section 10)
- **Rule-based baseline**: 51.4% alert rate, 90.7% recall, 0.16% precision — realistic of real-world AML alert fatigue
- **ACH, not Bitcoin/Cash**, is the strongest single laundering signal in this dataset — counter to conventional assumption
- **Cost-sensitive XGBoost**: ~5.2x fewer alerts than the baseline at default threshold (87.2% recall); ~2.1x fewer at matched recall
- **Cost-based evaluation** (using real transaction amounts): a more modest but still real 1.5-1.6x total cost reduction — smaller than the alert-volume figure, because ML's misses skew toward larger, non-ACH-format transactions
- **SHAP confirms the mechanism**: `amount_paid` actively works *against* detection for non-ACH laundering cases, explaining the cost-evaluation finding precisely

## Limitations (stated explicitly, not hidden)

- Single dataset variant (HI-Small) — see risk register / variant justification
- No hyperparameter tuning, no cross-validation, no bootstrap confidence intervals — time-constrained scope decision
- Cost model uses an illustrative per-alert review cost ($30) — adjust/cite from literature if a more precise figure is available
- Practitioner validation (Objective 7) — separate qualitative strand, tracked independently

## AI Declaration

AITS Level 2 (AI for Shaping) — outputs independently reviewed, critically assessed, and
substantially revised. Research design, analysis, and academic arguments are the author's own.
