import unittest
import pandas as pd

from detectors import category_amount


def toy():
    return pd.DataFrame({
        "category": ["food", "food", "food", "food", "travel", "travel", "food"],
        "amount":   [10.0, 20.0, 30.0, 40.0, 500.0, 1000.0, 9999.0],
        "fraud":    [0, 0, 0, 0, 0, 0, 1],
    })


class TestCategoryAmount(unittest.TestCase):

    def test_scores_between_0_and_1(self):
        scores = category_amount.score(toy())
        self.assertTrue(scores.between(0, 1).all())

    def test_largest_and_smallest_amounts(self):
        test = pd.DataFrame({"category": ["food", "food"], "amount": [50.0, 5.0]})
        scores = category_amount.score(test, toy())
        self.assertEqual(scores.tolist(), [1.0, 0.0])

    def test_percentile_within_category(self):
        test = pd.DataFrame({"category": ["food"], "amount": [20.0]})
        self.assertEqual(category_amount.score(test, toy()).iloc[0], 0.5)

    def test_fraud_rows_left_out_of_reference(self):
        test = pd.DataFrame({"category": ["food"], "amount": [5000.0]})
        self.assertEqual(category_amount.score(test, toy()).iloc[0], 1.0)

    def test_same_amount_scores_differently_by_category(self):
        test = pd.DataFrame({"category": ["food", "travel"], "amount": [500.0, 500.0]})
        scores = category_amount.score(test, toy())
        self.assertEqual(scores.tolist(), [1.0, 0.5])

    def test_bigger_amount_never_scores_lower(self):
        test = pd.DataFrame({"category": ["food"] * 5, "amount": [1.0, 15.0, 25.0, 35.0, 100.0]})
        scores = category_amount.score(test, toy())
        self.assertTrue(scores.is_monotonic_increasing)

    def test_unseen_category_compared_to_all_normal_payments(self):
        test = pd.DataFrame({"category": ["zzz"], "amount": [35.0]})
        self.assertEqual(category_amount.score(test, toy()).iloc[0], 0.5)

    def test_quoted_categories_match(self):
        test = pd.DataFrame({"category": ["'travel'"], "amount": [500.0]})
        self.assertEqual(category_amount.score(test, toy()).iloc[0], 0.5)

    def test_missing_column(self):
        with self.assertRaises(ValueError):
            category_amount.score(pd.DataFrame({"category": ["food"]}))


if __name__ == "__main__":
    unittest.main(verbosity=2)
