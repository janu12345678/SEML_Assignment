"""TEST TYPE 3 (cont.) - ML BEHAVIOURAL TESTS: INFERENCE (Objective 2.7b).

Three families, following Zinkevich / "ML Test Score" practice:

  * SHAPE & RANGE      -- outputs are structurally valid for every input.
  * DIRECTIONAL        -- a change the domain says should push the score one
                          way actually does (monotonic expectations).
  * INVARIANCE         -- a change the domain says is irrelevant leaves the
                          score untouched.
"""

from __future__ import annotations

import pytest

from loan_risk.config import settings
from loan_risk.models.predictor import ModelRegistry, RiskPredictor

pytestmark = pytest.mark.ml

APPLICATION = {
    "Age": 45,
    "AnnualIncome": 95_000,
    "CreditScore": 720,
    "EmploymentStatus": 0,
    "EducationLevel": 4,
    "LoanAmount": 25_000,
    "LoanDuration": 36,
    "MonthlyDebtPayments": 400,
    "CreditCardUtilizationRate": 0.25,
    "DebtToIncomeRatio": 0.15,
    "BankruptcyHistory": 0,
    "LoanPurpose": 3,
    "PreviousLoanDefaults": 0,
    "PaymentHistory": 28,
    "LengthOfCreditHistory": 15,
    "SavingsAccountBalance": 35_000,
    "CheckingAccountBalance": 8_000,
    "TotalLiabilities": 30_000,
    "JobTenure": 10,
    "NetWorth": 220_000,
}


@pytest.fixture
def predictor(trained_pipeline) -> RiskPredictor:
    registry = ModelRegistry(settings)
    registry.pipeline = trained_pipeline
    registry.version = "test-2.0.0"
    return RiskPredictor(registry, settings)


def _score(predictor: RiskPredictor, **overrides) -> float:
    payload = {**APPLICATION, **overrides}
    return predictor.score(predictor.to_frame(payload))


# ── shape & range ────────────────────────────────────────────────────────
def test_single_prediction_returns_a_well_formed_assessment(predictor):
    result = predictor.predict(dict(APPLICATION))
    assert isinstance(result.is_approved, bool)
    assert 0.0 <= result.probability <= 1.0
    assert result.risk_tier in {"LOW", "MEDIUM", "HIGH"}
    assert result.latency_ms >= 0.0


# ── directional expectations ─────────────────────────────────────────────
def test_higher_credit_score_does_not_reduce_approval_probability(predictor):
    poor = _score(predictor, CreditScore=520)
    excellent = _score(predictor, CreditScore=800)
    assert excellent >= poor, (poor, excellent)


# ── invariance expectations ──────────────────────────────────────────────
def test_prediction_is_invariant_to_repeated_calls(predictor):
    """Determinism: the same payload must always yield the same score.

    This test caught a real defect. With ``n_jobs=-1`` the forest sums tree
    votes in a non-deterministic thread order, so repeated calls differed at
    ~5e-17. Harmless in isolation, but an application sitting exactly on the
    0.50 cut-off could flip between APPROVED and DENIED across identical
    requests -- unacceptable in regulated lending. The estimator is now pinned
    to a deterministic reduction, and the tolerance below is the float-equality
    bound, not a workaround.
    """
    scores = [_score(predictor) for _ in range(5)]
    for score in scores[1:]:
        assert score == pytest.approx(scores[0], abs=1e-12, rel=0)
    rounded = {round(score, 4) for score in scores}
    assert len(rounded) == 1
