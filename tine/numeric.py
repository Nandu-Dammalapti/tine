"""Numeric conversion that rejects values whose meaning would be changed."""

import numpy as np


def real_array(values, name):
    """Convert numeric inputs without turning booleans or complex values into reals.

    Preserve element types for sequences before NumPy can promote mixed values.
    Already numeric arrays use a fast path without a Python loop.
    """
    message = f"{name} must contain real numeric values, excluding booleans and complex numbers"
    try:
        raw = np.asarray(values) if isinstance(values, np.ndarray) else np.asarray(values, dtype=object)
        if raw.dtype.kind in "bcmM":
            raise ValueError(message)
        if raw.dtype.kind == "O" and any(
            isinstance(value, (bool, np.bool_, complex, np.complexfloating))
            for value in raw.flat
        ):
            raise ValueError(message)
        return np.asarray(raw, dtype=float)
    except (TypeError, ValueError, OverflowError):
        raise ValueError(message) from None
