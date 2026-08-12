"""TEST TYPE 2 - DATA VALIDATION / DATA-QUALITY TESTS.

These assert on the *data contract* rather than on code behaviour. In an ML
system the data is as much a dependency as a library is, so it needs its own
regression suite: schema conformance, missing values and distribution drift.
"""

from __future__ import annotations

import numpy as np
import pytest

from loan_risk.data.validation import DataValidator

pytestmark = pytest.mark.data


# ── DQ metric 1: schema conformance ──────────────────────────────────────
def test_clean_data_conforms_fully(synthetic_frame):
    report = DataValidator().validate(synthetic_frame)
    assert report.passed
    assert report.schema_conformance_rate == 1.0


def test_out_of_range_credit_score_is_flagged(synthetic_frame):
    bad = synthetic_frame.copy()
    bad.loc[bad.index[0], "CreditScore"] = 9999
    report = DataValidator().validate(bad)
    assert "above_maximum:CreditScore" in report.violations
    assert report.schema_conformance_rate < 1.0


# ── DQ metric 2: missing values ──────────────────────────────────────────
def test_missing_value_fraction_is_measured(synthetic_frame):
    dirty = synthetic_frame.copy().astype({"PaymentHistory": "float"})
    # Null out a *fraction* of rows, not a fixed count, so the assertion holds
    # whatever size the fixture frame is.
    n_null = int(len(dirty) * 0.20)
    dirty.loc[dirty.index[:n_null], "PaymentHistory"] = np.nan
    report = DataValidator(max_missing_fraction=0.001).validate(dirty)
    assert report.missing_value_fraction > 0
    assert report.worst_column == "PaymentHistory"
    assert any(v.startswith("missing_fraction_exceeded") for v in report.violations)
    assert "nulls_in_non_nullable:PaymentHistory" in report.violations
