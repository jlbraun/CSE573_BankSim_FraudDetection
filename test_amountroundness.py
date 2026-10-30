import unittest
import pandas as pd

from detectors import amount_roundness


class TestAmountRoundness(unittest.TestCase):

    def test_whole_amounts(self):
        df = pd.DataFrame({"amount": [1.00, 5.00, 10.00, 100.00]})
        self.assertEqual(
            amount_roundness.score(df).tolist(),
            [1.0, 1.0, 1.0, 1.0],
        )

    def test_amounts_with_cents(self):
        df = pd.DataFrame({"amount": [1.01, 4.99, 47.83, 100.50]})
        self.assertEqual(
            amount_roundness.score(df).tolist(),
            [0.0, 0.0, 0.0, 0.0],
        )

    def test_zero_amount(self):
        df = pd.DataFrame({"amount": [0.00]})
        self.assertEqual(amount_roundness.score(df).tolist(), [0.0])

    def test_missing_column(self):
        df = pd.DataFrame({"customer": ["A"]})
        with self.assertRaises(ValueError):
            amount_roundness.score(df)

    def test_invalid_amounts(self):
        for value in [-5, None, float("inf"), "invalid"]:
            with self.subTest(amount=value):
                df = pd.DataFrame({"amount": [value]})
                with self.assertRaises(ValueError):
                    amount_roundness.score(df)

    def test_preserves_index_and_input(self):
        df = pd.DataFrame(
            {"amount": [10.00, 12.34, 5.00]},
            index=[8, 2, 8],
        )
        original = df.copy(deep=True)

        result = amount_roundness.score(df)

        self.assertEqual(result.tolist(), [1.0, 0.0, 1.0])
        self.assertTrue(result.index.equals(df.index))
        pd.testing.assert_frame_equal(df, original)

    def test_small_rounding_difference(self):
        df = pd.DataFrame({"amount": [10.000000001, 10.01]})
        self.assertEqual(
            amount_roundness.score(df).tolist(),
            [1.0, 0.0],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)