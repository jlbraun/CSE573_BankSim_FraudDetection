# Detectors — Mansi Ganga Nayak

All three return a score between 0 and 1 per transaction and live in `detectors/`.

| Detector | File | Signal | Test accuracy (50/50 split) | ROC-AUC |
|---|---|---|---|---|
| Category Risk | `category_risk.py` | fraud rate of the merchant category, learned from train labels | 0.953 @ 0.05 | 0.98 |
| Category Novelty | `category_novelty.py` | customer buying in a category they rarely/never use, weighted by how much history they have | 0.877 @ 0.3 (0.524 @ 0.05) | 0.86 |
| Time Gap | `time_gap.py` | days since the customer's previous transaction (dormant account coming back) | 0.662 @ 0.05 | 0.67 |

## Usage
```python
category_risk.score(test, train)          # needs labeled train split
time_gap.score(test, history)             # history = full cleaned bs140513_032310.csv
category_novelty.score(test, history)
```
`time_gap` and `category_novelty` use **no fraud labels** and only look at each customer's
**earlier** transactions, so there is no leakage. Pass the full dataset as `history`: the
downsampled files keep only a few rows per customer, and without the full history both detectors
drop to ~0.70 (novelty) and ~0.54 (time gap) ROC-AUC.

## Notes for the ensemble
- `main.py`'s `acc()` uses a fixed 0.05 cutoff. That suits Category Risk and Time Gap, but
  Category Novelty's best cutoff is ~0.3 (0.877 accuracy). A per-detector threshold or the
  score-band weighting would use it properly.
- Category Novelty: usual category (>50% of history) ~0.03% fraud; rare/new category ~4–9% fraud.
- Time Gap: gap 1 day ~0.8% fraud; gap 3+ days ~15–23% fraud. Weak alone, useful in the ensemble.
- Tests: `python test_category_risk.py`, `python test_time_gap.py`, `python test_category_novelty.py`
