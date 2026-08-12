"""TEST TYPE 1 - UNIT TESTS.

Scope: one function or class at a time, no I/O, no model, no HTTP. These are
the fastest layer of the pyramid and are what makes refactoring safe.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from loan_risk.config import settings
from loan_risk.exceptions import FeatureEngineeringError
from loan_risk.features.engineering import (
    FEATURE_ORDER,
    FeatureEngineer,
    compute_loan_to_income,
    safe_ratio,
)

pytestmark = pytest.mark.unit


# ── pure helper functions ────────────────────────────────────────────────
def test_safe_ratio_never_divides_by_zero():
    """The +1 smoothing must make a zero denominator harmless."""
    assert safe_ratio(50.0, 0.0) == pytest.approx(50.0)
    assert np.isfinite(safe_ratio(1.0, 0.0))


def test_safe_ratio_produces_correct_value_for_normal_inputs():
    """safe_ratio(a, b) must equal a / (b + 1) for positive inputs."""
    # 50000 / (100000 + 1) = 0.49999500...
    result = safe_ratio(50000.0, 100000.0)
    assert result == pytest.approx(50000.0 / 100001.0)


def test_compute_loan_to_income_returns_expected_value():
    """Leverage ratio computation must use safe_ratio internally."""
    # loan=50000, income=100000 -> 50000 / (100000 + 1) ~ 0.49999
    result = compute_loan_to_income(50000, 100000)
    assert result == pytest.approx(50000 / 100001, rel=1e-4)


# ── FeatureEngineer transformer ──────────────────────────────────────────
def test_transformer_emits_the_declared_feature_contract(synthetic_frame):
    """Output columns must match FEATURE_ORDER exactly."""
    engineer = FeatureEngineer()
    out = engineer.fit_transform(synthetic_frame)
    assert list(out.columns) == FEATURE_ORDER
    assert out.shape == (len(synthetic_frame), settings.model.n_features)


def test_feature_engineer_rejects_non_dataframe_input():
    """Passing a non-DataFrame (e.g., a numpy array) must raise FeatureEngineeringError."""
    engineer = FeatureEngineer()
    with pytest.raises(FeatureEngineeringError, match="Expected a pandas DataFrame"):
        engineer.transform(np.array([[1, 2, 3]]))
