"""Tests for the StockMetrics analysis pipeline.

Run with: pytest
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.constants import EXPECTED_STOCK_CODE_TO_NAME
from src.utils.data_loader import derive_code_to_name
from src.analysis.analysis import (
    _permutation_pvalue,
    run_regression_analysis,
    run_comovement_analysis,
)


def _fake_sheets():
    """Stand-ins for the two data.xls sheets (all five firms), joinable on ISIN."""
    rows = [
        ("B01NPJ", "INE467B01029", "Tata Consultancy Services Ltd."),
        ("620512", "INE009A01021", "Infosys Ltd."),
        ("629489", "INE860A01027", "HCL Technologies Ltd."),
        ("620605", "INE075A01022", "Wipro Ltd."),
        ("BWFGD6", "INE669C01036", "Tech Mahindra Ltd."),
    ]
    stock = pd.DataFrame({"Symbol": ["Name", "ISIN Number", "2020-12-31", "2021-12-31"]})
    for code, isin, _name in rows:
        stock[code] = [code, isin, 100.0, 110.0]
    fund = pd.DataFrame({
        "ISIN": [isin for _c, isin, _n in rows],
        "Company name": [name for _c, _i, name in rows],
        "Field": ["SALES"] * len(rows),
    })
    return stock, fund


def test_derive_code_to_name_uses_isin_join():
    stock, fund = _fake_sheets()
    mapping = derive_code_to_name(stock, fund)
    assert mapping["B01NPJ"] == "Tata Consultancy Services Ltd."
    assert mapping["620512"] == "Infosys Ltd."


def test_derive_code_to_name_raises_on_expected_mismatch(monkeypatch):
    stock, fund = _fake_sheets()
    # Tamper the expectation so the file-derived mapping disagrees with it.
    monkeypatch.setitem(EXPECTED_STOCK_CODE_TO_NAME, "B01NPJ", "Wrong Company")
    with pytest.raises(ValueError):
        derive_code_to_name(stock, fund)


def test_permutation_pvalue_high_for_noise():
    rng = np.random.default_rng(0)
    n = 30
    X = sm.add_constant(rng.normal(size=(n, 3)))
    y = rng.normal(size=n)  # no relationship to X
    observed = sm.OLS(y, X).fit().rsquared
    p = _permutation_pvalue(X, y, observed, n_permutations=300, seed=1)
    # Noise: the observed R^2 sits in the middle of the null, so p should be large.
    assert p > 0.2


def test_permutation_pvalue_low_for_strong_signal():
    rng = np.random.default_rng(0)
    n = 60
    x = rng.normal(size=n)
    X = sm.add_constant(np.column_stack([x, rng.normal(size=n)]))
    y = 3.0 * x + rng.normal(size=n) * 0.1  # strong relationship
    observed = sm.OLS(y, X).fit().rsquared
    p = _permutation_pvalue(X, y, observed, n_permutations=300, seed=1)
    assert p < 0.01


def test_comovement_detects_real_signal_and_ignores_noise():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2005-12-31", periods=17, freq="YE")
    # Two correlated "stocks" sharing a common factor, plus one independent.
    factor = rng.normal(size=17)
    code_to_name = {"A": "Alpha", "B": "Beta", "C": "Gamma"}
    returns = pd.DataFrame({
        "A": factor + rng.normal(scale=0.2, size=17),
        "B": factor + rng.normal(scale=0.2, size=17),
        "C": rng.normal(size=17),
    }, index=dates)

    rows = run_comovement_analysis(returns, code_to_name)
    by_pair = {frozenset((r["company_a"], r["company_b"])): r for r in rows}
    ab = by_pair[frozenset(("Alpha", "Beta"))]
    assert ab["correlation"] > 0.8 and ab["significant"]   # real co-movement found
    ac = by_pair[frozenset(("Alpha", "Gamma"))]
    assert not ac["significant"]                            # noise not flagged


def test_regression_reports_no_significant_after_fdr():
    """On the real-style tiny-n regime, nothing should survive FDR correction."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2005-12-31", periods=17, freq="YE")
    code_to_name = {"B01NPJ": "Firm A"}
    stock = pd.DataFrame({"B01NPJ": rng.normal(size=17)}, index=dates)
    fund_metrics = {
        f"m{i}": pd.DataFrame({"Firm A": rng.normal(size=17)}, index=dates)
        for i in range(5)
    }
    results, top3, _ = run_regression_analysis(stock, fund_metrics, code_to_name)
    assert results["Firm A"]["significant_vars"] == []
    assert top3 == []
    assert 0.0 <= results["Firm A"]["perm_pvalue_r2"] <= 1.0
