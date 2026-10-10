import unittest
import pandas as pd

from detectors import velocity


class TestVelocity(unittest.TestCase):

    def test_previous_step_counts(self):
        df = pd.DataFrame({
            "customer": ["A", "A", "A", "A", "B", "B"],
            "step": [0, 0, 1, 2, 0, 1],
        })

        actual = velocity.counts(df).tolist()

        self.assertEqual(actual, [0, 0, 2, 1, 0, 1])

    def test_current_step_is_excluded(self):
        df = pd.DataFrame({
            "customer": ["A", "A", "A"],
            "step": [1, 1, 1],
        })

        self.assertEqual(velocity.counts(df).tolist(), [0, 0, 0])

    def test_score_calculation(self):
        df = pd.DataFrame({
            "customer": ["A"] * 6,
            "step": [0, 0, 0, 0, 0, 1],
        })

        scores = velocity.score(df)

        self.assertAlmostEqual(scores.iloc[-1], 0.5)
        self.assertTrue(scores.between(0, 1).all())

    def test_unsorted_rows(self):
        df = pd.DataFrame({
            "customer": ["A", "A", "A"],
            "step": [1, 0, 0],
        })

        self.assertEqual(velocity.counts(df).tolist(), [2, 0, 0])

    def test_missing_column(self):
        df = pd.DataFrame({"customer": ["A"]})

        with self.assertRaises(ValueError):
            velocity.score(df)


if __name__ == "__main__":
    unittest.main(verbosity=2)