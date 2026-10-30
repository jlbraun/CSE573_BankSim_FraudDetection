import numpy as np
import pandas as pd


def score(df):
    """Flag positive whole-number payment amounts.

    Returns 1.0 for round amounts and 0.0 otherwise.
    This is an indicator, not a fraud probability.
    """
    if "amount" not in df.columns:
        raise ValueError("Missing required column: amount")

    amounts = pd.to_numeric(df["amount"], errors="raise")

    if amounts.isna().any() or not np.isfinite(amounts).all():
        raise ValueError("Amounts must be finite numbers without missing values")

    if (amounts < 0).any():
        raise ValueError("Amounts cannot be negative")

    # A round payment is positive and has no fractional part.
    whole_number = np.isclose(
        amounts,
        np.round(amounts),
        rtol=0,
        atol=1e-8,
    )

    round_payment = (amounts > 0) & whole_number

    return pd.Series(
        round_payment.astype(float),
        index=df.index,
        name="score_amount_roundness",
    )