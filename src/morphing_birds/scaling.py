"""Unit conversion utilities."""

from __future__ import annotations

# Common length unit factors (relative to metres)
UNIT_FACTORS: dict[str, float] = {
    "mm": 0.001,
    "cm": 0.01,
    "m": 1.0,
}


def unit_conversion_factor(from_unit: str, to_unit: str) -> float:
    """Return the multiplicative factor to convert *from_unit* to *to_unit*.

    Parameters
    ----------
    from_unit, to_unit : str
        Unit strings — one of ``'mm'``, ``'cm'``, ``'m'``.

    Returns
    -------
    float
        The conversion factor.

    Raises
    ------
    ValueError
        If either unit is not recognised.
    """
    if from_unit not in UNIT_FACTORS:
        msg = f"Unknown unit '{from_unit}'. Known: {list(UNIT_FACTORS)}"
        raise ValueError(msg)
    if to_unit not in UNIT_FACTORS:
        msg = f"Unknown unit '{to_unit}'. Known: {list(UNIT_FACTORS)}"
        raise ValueError(msg)
    return UNIT_FACTORS[from_unit] / UNIT_FACTORS[to_unit]
