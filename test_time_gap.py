import unittest
import pandas as pd

from detectors import time_gap


class TestTimeGap(unittest.TestCase):

    def test_gap_days(self):
        df = pd.DataFrame({"customer": ["A", "A", "A", "A", "B"],
                           "step":     [0,   1,   1,   5,   3]})
        self.assertEqual(time_gap.gaps(df).fillna(-1).tolist(), [-1, 1, 1, 4, -1])

    def test_daily_customer_scores_zero(self):
        df = pd.DataFrame({"customer": ["A"] * 4, "step": [0, 1, 2, 3]})
        self.assertEqual(time_gap.score(df).tolist()[1:], [0.0, 0.0, 0.0])

    def test_longer_gap_scores_higher(self):
        df = pd.DataFrame({"customer": ["A", "A", "B", "B"], "step": [0, 3, 0, 10]})
        s = time_gap.score(df)
        self.assertAlmostEqual(s.iloc[1], 0.5)         # gap 3, scale 2 -> 2 / (2 + 2)
        self.assertTrue(s.iloc[3] > s.iloc[1])
        self.assertTrue(s.between(0, 1).all())

    def test_first_transaction_gets_first_score(self):
        df = pd.DataFrame({"customer": ["A"], "step": [4]})
        self.assertEqual(time_gap.score(df, first_score=0.3).iloc[0], 0.3)

    def test_history_fills_gaps_in_a_sample(self):
        history = pd.DataFrame({"customer": ["A"] * 4, "step": [0, 1, 2, 3]})
        sample = history.iloc[[0, 3]]                  # sample skipped days 1 and 2
        self.assertEqual(time_gap.gaps(sample, history).fillna(-1).tolist(), [-1, 1])
        self.assertEqual(time_gap.gaps(sample).fillna(-1).tolist(), [-1, 3])

    def test_row_not_in_history_uses_earlier_days(self):
        history = pd.DataFrame({"customer": ["A", "A"], "step": [0, 2]})
        new = pd.DataFrame({"customer": ["A"], "step": [6]})
        self.assertEqual(time_gap.gaps(new, history).iloc[0], 4)

    def test_quoted_values_and_missing_columns(self):
        df = pd.DataFrame({"customer": ["'A'", "A"], "step": [0, 1]})
        self.assertEqual(time_gap.gaps(df).iloc[1], 1)
        with self.assertRaises(ValueError):
            time_gap.score(pd.DataFrame({"customer": ["A"]}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
