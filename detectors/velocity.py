# Transaction Velocity Detector
# Author: Hira Naseer

import numpy as np
import pandas as pd


def counts(df, window_steps=1):
    """Count each customer's payments in earlier time steps."""

    if (
        isinstance(window_steps, bool)
        or not isinstance(window_steps, (int, np.integer))
        or window_steps < 1
    ):
        raise ValueError("window_steps must be a positive integer")

    required = {"customer", "step"}
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    if df["customer"].isna().any() or df["step"].isna().any():
        raise ValueError("customer and step cannot contain missing values")

    steps = pd.to_numeric(df["step"], errors="raise").to_numpy(dtype=float)

    if (
        not np.isfinite(steps).all()
        or (steps < 0).any()
        or (steps != np.floor(steps)).any()
    ):
        raise ValueError("step must contain nonnegative whole numbers")

    work = pd.DataFrame({
        "customer": df["customer"].to_numpy(),
        "step": steps,
        "position": np.arange(len(df)),
    })

    result = np.zeros(len(df), dtype=np.int64)

    for _, group in work.groupby("customer", sort=False):
        # Number of payments at each step for this customer.
        payments = group.groupby("step").size().sort_index()
        times = payments.index.to_numpy()

        # Cumulative counts let us count a historical window efficiently.
        cumulative = np.r_[0, np.cumsum(payments.to_numpy())]
        current_steps = group["step"].to_numpy()

        start = np.searchsorted(
            times, current_steps - window_steps, side="left"
        )
        end = np.searchsorted(times, current_steps, side="left")

        result[group["position"].to_numpy()] = (
            cumulative[end] - cumulative[start]
        )

    return pd.Series(
        result,
        index=df.index,
        name="transaction_velocity_count",
    )


def score(df, window_steps=1, scale=5.0):
    """Return a velocity score between 0 and 1."""

    if isinstance(scale, bool) or not np.isscalar(scale):
        raise ValueError("scale must be a positive number")

    scale = float(scale)

    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("scale must be a finite positive number")

    payment_counts = counts(df, window_steps).astype(float)

    return (
        payment_counts / (payment_counts + scale)
    ).rename("score_velocity")