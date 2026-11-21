# Detector analysis & tuning — Mansi Ganga Nayak

Covers the three pre-demo tasks assigned on Nov 13:
1. Why do `merchant_risk` and `category_risk` give almost the same output?
2. Why does `merchant_risk` dominate the ensemble's weight?
3. Tune hyperparameters of the detectors and the ensemble.

All numbers can be reproduced with the scripts in `analysis/` (raw tables in `analysis/results/`).

## How we evaluated (and why)
The professor pointed out that accuracy is a vanity metric at 1.21% fraud. So besides the
50/50 test split, everything here is also measured on a **realistic test set**: every row of
`bs140513_032310.csv` that is *not* in the 50/50 training split (583,123 payments, **0.25% fraud**,
because most fraud rows went into training). That set is split 50/50 into a **validation half**
(used for every tuning decision) and a **test half** (reported once, never tuned on).
Main metric: **PR-AUC on the fraud class**, plus precision/recall/F1.

## 1. merchant_risk vs category_risk — `analysis/merchant_vs_category.py`
* **All 50 merchants sell in exactly one category.** A merchant is a finer split of a category.
* So a merchant's fraud rate tracks its category's: the two scores correlate at **r = 0.97**.
* At the 0.05 cutoff they flag **exactly the same rows (100%)**: zero-fraud categories
  (food, transportation, contents) fall below 0.05, everything else is above. That is why
  their accuracy (95.28%) and confusion matrices were identical.
* They are not the same score: inside a category merchants differ a lot (es_health: 0%, 0%, 2%, 5%, **47%**;
  es_hotelservices: 0% to 86%), so `merchant_risk` holds everything `category_risk` knows **plus more**.
* This is also why the "merchant category mismatch" detector was dropped (it would always be 0).

## 2. Why merchant_risk dominates — `analysis/dominance_experiments.py`
| Setup | GB weight of merchant_risk | PR-AUC 50/50 | PR-AUC realistic |
|---|---|---|---|
| A. current main.py | 0.960 | 0.997 | 0.698 |
| B. label detectors computed out-of-fold (no label leakage) | 0.957 | 0.997 | 0.693 |
| C. drop category_risk | 0.961 | 0.998 | 0.691 |
| E. drop merchant_risk | — (category_risk takes 0.915) | 0.993 | **0.535** |

* **Not label leakage**: computing the risk scores out-of-fold changes nothing.
* **Not caused by category_risk**: removing it changes nothing (it is redundant, see task 1).
* **It is a genuinely strong signal**: removing merchant_risk drops realistic PR-AUC from 0.70 to 0.54,
  and category_risk simply takes its place. Permutation importance (a fairer measure for correlated
  features) agrees: merchant_risk 0.22, category_amount 0.02, everything else ≈ 0.
* **Conclusion to document:** in BankSim, fraud is concentrated in a few merchants, so a merchant's past
  fraud rate is by far the best single predictor. The dominance reflects the synthetic data, not a bug.
  The other detectors only add value *within* risky merchants (category_amount helps most).
  Spreading the weight would need features that carry information the merchant doesn't; that is
  where the graph features the professor suggested (degree, centrality, community) could help.

## 3. Tuning — `analysis/tune.py`, `analysis/operating_points.py`
Gradient boosting is unaffected by monotonic rescaling, so detector "scale" settings change nothing in the
ensemble. Only settings that change the *ordering* of scores were tuned (one at a time, on validation):

| Setting | Tried | Chosen | Note |
|---|---|---|---|
| merchant_risk smoothing | 0, 5, 20, 50 | **5** (was 20) | |
| category_risk smoothing | 0, 5, 20, 50 | **5** (was 20) | |
| category_novelty k | 0, 2, 5, 10 | **0** (was 2) | |
| time_gap first_score | 0, 0.5, 1 | 0.5 (unchanged) | |
| velocity window_steps | 1, 2, 3, 7 | 1 (unchanged) | longer windows hurt |
| GB max_depth / learning_rate | 2–6 / 0.03–0.1 (+ n_estimators, subsample, min_samples_leaf: 64 combos) | **6 / 0.03** (was 4 / 0.05) | |

**Held-out realistic test (291,562 payments, 720 fraud):**

| Setup | Precision | Recall | F1 | Alerts | PR-AUC |
|---|---|---|---|---|---|
| current main.py, threshold 0.05 | 4.8% | 100% | 0.09 | 15,116 | 0.702 |
| current main.py, threshold 0.5 | 8.4% | 96.9% | 0.15 | 8,295 | 0.702 |
| **tuned, threshold 0.966 (max F1 on validation)** | **67.4%** | 65.7% | **0.67** | 702 | **0.711** |

The setting changes give a small gain (PR-AUC 0.702 → 0.711). **The big win is the decision threshold**:
the model is trained on 50/50 data, so on real data its probabilities are far too high and the cutoff
must be raised. Choose the operating point by how many missed frauds vs false alerts we accept:

| Rule (chosen on validation) | Threshold | Test precision | Test recall | Caught / missed | False alerts |
|---|---|---|---|---|---|
| recall ≥ 99% | 0.342 | 6.8% | 98.1% | 706 / 14 | 9,700 |
| recall ≥ 95% | 0.781 | 13.1% | 93.6% | 674 / 46 | 4,472 |
| recall ≥ 90% | 0.898 | 29.7% | 89.2% | 642 / 78 | 1,517 |
| recall ≥ 80% | 0.948 | 48.9% | 80.6% | 580 / 140 | 606 |
| max F1 | 0.966 | 67.4% | 65.7% | 473 / 247 | 229 |

**Per-detector thresholds** (chosen on the 50/50 train split, used in main.py's accuracy table):

| Detector | Old acc (cut .05) | Tuned cut | New acc |
|---|---|---|---|
| amount_anomaly | 57.8% | 0.035 | 61.0% |
| merchant_risk | 95.3% | 0.417 | 97.5% |
| category_amount | 52.5% | 0.867 | 83.0% |
| category_risk | 95.3% | 0.227 | 95.4% |
| velocity | 52.2% | 0.05 (kept) | 52.2% |
| merchant_novelty | 80.1% | 0.143 | 86.0% |
| time_gap | 66.2% | 0.333 | 66.2% |
| category_novelty | 52.4% | 0.5 | 89.1% |
| ensemble (50/50 split) | 97.57% | 0.05 (kept) | 97.50% |

The 50/50 ensemble accuracy is saturated (≈97.5% either way); the tuning targets the realistic data.

## Changes in main.py (branch `tuning-merchant-dominance`)
* `merchant_risk` / `category_risk` smoothing 20 → 5, `category_novelty` k 2 → 0.
* Gradient boosting max_depth 4 → 6, learning_rate 0.05 → 0.03.
* Per-detector thresholds (`THRESHOLDS`) in the accuracy / confusion table; ensemble stays at 0.05.
* Nothing removed; the realistic-data evaluation lives in `analysis/` so it does not clash with the
  full-dataset testing being added to main.py.

## Suggested next steps
* Use one of the operating points above when main.py is tested on the full CSV.
* Graph features (degree / centrality / community of customers and merchants) are the most promising
  way to add information the merchant doesn't already carry.
