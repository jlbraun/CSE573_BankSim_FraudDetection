import unittest
import pandas as pd

from detectors import merchant_risk


def toy():
    return pd.DataFrame({
        "merchant": ["m1", "m1", "m1", "m2", "m2", "m2", "m3", "m3"],
        "fraud":    [1, 1, 1, 0, 0, 0, 1, 0],
    })


class TestMerchantRisk(unittest.TestCase):

    def test_scores_between_0_and_1(self):
        scores = merchant_risk.score(toy())
        self.assertTrue(scores.between(0, 1).all())

    def test_no_smoothing_gives_exact_rates(self):
        scores = merchant_risk.score(toy(), smoothing=0)
        self.assertEqual(scores.tolist(), [1, 1, 1, 0, 0, 0, 0.5, 0.5])

    def test_smoothing_pulls_toward_overall_rate(self):
        overall = toy()["fraud"].mean()
        scores = merchant_risk.score(toy(), smoothing=10)
        self.assertTrue(overall < scores.iloc[0] < 1)   # risky merchant m1
        self.assertTrue(0 < scores.iloc[3] < overall)   # safe merchant m2

    def test_test_rows_use_train_rates_not_their_own_labels(self):
        test = pd.DataFrame({"merchant": ["m1", "m2"], "fraud": [0, 1]})
        scores = merchant_risk.score(test, toy(), smoothing=0)
        self.assertEqual(scores.tolist(), [1.0, 0.0])

    def test_unseen_merchant_gets_overall_rate(self):
        test = pd.DataFrame({"merchant": ["zzz"]})
        scores = merchant_risk.score(test, toy())
        self.assertAlmostEqual(scores.iloc[0], toy()["fraud"].mean())

    def test_quoted_merchants_match(self):
        test = pd.DataFrame({"merchant": ["'m1'"]})
        self.assertEqual(merchant_risk.score(test, toy(), smoothing=0).iloc[0], 1.0)

    def test_missing_column(self):
        with self.assertRaises(ValueError):
            merchant_risk.score(pd.DataFrame({"amount": [1.0]}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
