import unittest
import pandas as pd

from detectors import category_risk


def toy():
    return pd.DataFrame({
        "category": ["a", "a", "a", "b", "b", "b", "c", "c"],
        "fraud":    [1, 1, 1, 0, 0, 0, 1, 0],
    })


class TestCategoryRisk(unittest.TestCase):

    def test_scores_between_0_and_1(self):
        scores = category_risk.score(toy())
        self.assertTrue(scores.between(0, 1).all())

    def test_no_smoothing_gives_exact_rates(self):
        scores = category_risk.score(toy(), smoothing=0)
        self.assertEqual(scores.tolist(), [1, 1, 1, 0, 0, 0, 0.5, 0.5])

    def test_smoothing_pulls_toward_overall_rate(self):
        overall = toy()["fraud"].mean()
        scores = category_risk.score(toy(), smoothing=10)
        self.assertTrue(overall < scores.iloc[0] < 1)   # risky category a
        self.assertTrue(0 < scores.iloc[3] < overall)   # safe category b

    def test_test_rows_use_train_rates_not_their_own_labels(self):
        test = pd.DataFrame({"category": ["a", "b"], "fraud": [0, 1]})
        scores = category_risk.score(test, toy(), smoothing=0)
        self.assertEqual(scores.tolist(), [1.0, 0.0])

    def test_unseen_category_gets_overall_rate(self):
        test = pd.DataFrame({"category": ["zzz"]})
        scores = category_risk.score(test, toy())
        self.assertAlmostEqual(scores.iloc[0], toy()["fraud"].mean())

    def test_quoted_categories_match(self):
        test = pd.DataFrame({"category": ["'a'"]})
        self.assertEqual(category_risk.score(test, toy(), smoothing=0).iloc[0], 1.0)

    def test_missing_column(self):
        with self.assertRaises(ValueError):
            category_risk.score(pd.DataFrame({"amount": [1.0]}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
