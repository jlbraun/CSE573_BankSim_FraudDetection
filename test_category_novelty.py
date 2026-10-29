import unittest
import pandas as pd

from detectors import category_novelty


class TestCategoryNovelty(unittest.TestCase):

    def test_counts_only_earlier_days(self):
        df = pd.DataFrame({"customer": ["A", "A", "A", "A"],
                           "category": ["food", "food", "food", "travel"],
                           "step":     [0, 1, 1, 2]})
        c = category_novelty.history_counts(df)
        self.assertEqual(c["prior_total"].tolist(), [0, 1, 1, 3])
        self.assertEqual(c["prior_cat"].tolist(), [0, 1, 1, 0])

    def test_usual_category_low_new_category_high(self):
        df = pd.DataFrame({"customer": ["A"] * 6,
                           "category": ["food"] * 5 + ["travel"],
                           "step":     [0, 1, 2, 3, 4, 5]})
        s = category_novelty.score(df, k=0)
        self.assertEqual(s.iloc[4], 0.0)               # 5th food purchase
        self.assertEqual(s.iloc[5], 1.0)               # first travel purchase after 5 food
        self.assertTrue(s.between(0, 1).all())

    def test_longer_history_is_more_suspicious(self):
        short = pd.DataFrame({"customer": ["A"] * 2, "category": ["food", "travel"], "step": [0, 1]})
        long = pd.DataFrame({"customer": ["B"] * 21, "category": ["food"] * 20 + ["travel"],
                             "step": list(range(21))})
        self.assertGreater(category_novelty.score(long).iloc[-1],
                           category_novelty.score(short).iloc[-1])

    def test_no_history_scores_zero(self):
        df = pd.DataFrame({"customer": ["A"], "category": ["travel"], "step": [0]})
        self.assertEqual(category_novelty.score(df).iloc[0], 0.0)

    def test_history_is_used_for_a_sample(self):
        history = pd.DataFrame({"customer": ["A"] * 4,
                                "category": ["food", "food", "food", "travel"],
                                "step": [0, 1, 2, 3]})
        sample = history.iloc[[3]]
        self.assertEqual(category_novelty.score(sample, history, k=0).iloc[0], 1.0)
        self.assertEqual(category_novelty.score(sample, k=0).iloc[0], 0.0)

    def test_row_not_in_history(self):
        history = pd.DataFrame({"customer": ["A", "A"], "category": ["food", "food"], "step": [0, 1]})
        new = pd.DataFrame({"customer": ["A"], "category": ["travel"], "step": [5]})
        c = category_novelty.history_counts(new, history)
        self.assertEqual((c["prior_total"].iloc[0], c["prior_cat"].iloc[0]), (2, 0))

    def test_missing_column(self):
        with self.assertRaises(ValueError):
            category_novelty.score(pd.DataFrame({"customer": ["A"], "step": [0]}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
