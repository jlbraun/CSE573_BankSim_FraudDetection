import unittest
import pandas as pd

from detectors import merchant_novelty


def toy():
    return pd.DataFrame({
        "customer": ["C1", "C1", "C1", "C2", "C2"],
        "merchant": ["m1", "m1", "m1", "m1", "m2"],
        "step":     [0, 1, 2, 3, 3],
    })


class TestMerchantNovelty(unittest.TestCase):

    def test_scores_between_0_and_1(self):
        scores = merchant_novelty.score(toy())
        self.assertTrue(scores.between(0, 1).all())

    def test_repeat_purchases_lower_the_score(self):
        scores = merchant_novelty.score(toy())
        self.assertEqual(scores.iloc[0], 1.0)
        self.assertEqual(scores.iloc[1], 0.5)
        self.assertAlmostEqual(scores.iloc[2], 1 / 3)

    def test_same_step_purchases_not_counted(self):
        df = pd.DataFrame({"customer": ["C1"] * 3, "merchant": ["m1"] * 3, "step": [5, 5, 5]})
        self.assertEqual(merchant_novelty.counts(df).tolist(), [0, 0, 0])

    def test_other_customers_at_same_merchant_dont_count(self):
        # C2 buys from m1 at step 3 after C1 bought there 3 times: still C2's first time
        self.assertEqual(merchant_novelty.score(toy()).iloc[3], 1.0)

    def test_history_supplies_earlier_purchases(self):
        df = pd.DataFrame({"customer": ["C1"], "merchant": ["m1"], "step": [10]})
        self.assertEqual(merchant_novelty.counts(df, toy()).tolist(), [3])
        self.assertEqual(merchant_novelty.counts(df).tolist(), [0])

    def test_unsorted_rows_keep_index(self):
        df = toy().iloc[[2, 0, 1]].set_index(pd.Index([10, 20, 30]))
        counts = merchant_novelty.counts(df)
        self.assertEqual(counts.tolist(), [2, 0, 1])
        self.assertEqual(counts.index.tolist(), [10, 20, 30])

    def test_quoted_ids_match(self):
        df = pd.DataFrame({"customer": ["'C1'"], "merchant": ["'m1'"], "step": [10]})
        self.assertEqual(merchant_novelty.counts(df, toy()).iloc[0], 3)

    def test_scale(self):
        scores = merchant_novelty.score(toy(), scale=2)
        self.assertAlmostEqual(scores.iloc[1], 2 / 3)
        with self.assertRaises(ValueError):
            merchant_novelty.score(toy(), scale=0)

    def test_missing_column(self):
        with self.assertRaises(ValueError):
            merchant_novelty.score(pd.DataFrame({"customer": ["C1"], "step": [0]}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
