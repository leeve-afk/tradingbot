"""Tests for aurum.strategies.ml_mlp (``ml_mlp``: ml_gbm's pipeline with an MLP ensemble).

Mirrors the ml_gbm tests that matter for a new model family: registry and option checks,
point-in-time discipline (perturbing/truncating the future leaves past forecasts
unchanged), determinism, a random-walk negative control, positive controls with planted
signals, and the epoch-wise early stopping on the time-ordered validation tail. Networks
are tiny to stay fast.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from test_strategies_ml import perturb_future, planted_market, sharpe_tstat

from aurum.core.types import MarketData
from aurum.data.synthetic import make_synthetic_bars
from aurum.labels import ewm_vol, fixed_horizon_labels, triple_barrier_labels
from aurum.strategies.base import _STRATEGIES, get_strategy
from aurum.strategies.ml import MLGBMStrategy
from aurum.strategies.ml_mlp import CalibratedMLP, MLMLPStrategy

GROUPS = ["returns", "trend", "meanrev"]
FAST = {"hidden_layer_sizes": (16,), "max_epochs": 40, "patience": 8, "n_ensemble": 2}
COMMON = dict(feature_groups=GROUPS, model=FAST, importance_repeats=1, importance_max_rows=400,
              n_threads=2, seed=0)


def test_registered_subclass_and_untrained_generate_raises():
    assert _STRATEGIES["ml_mlp"] is MLMLPStrategy
    assert issubclass(MLMLPStrategy, MLGBMStrategy)
    s = get_strategy("ml_mlp")
    assert s.trainable and not s.is_fitted
    assert s.default_params() == MLGBMStrategy.default_params()   # same knobs as ml_gbm
    with pytest.raises(RuntimeError):
        s.generate(MarketData(bars=make_synthetic_bars(50, "H1")))
    with pytest.raises(ValueError, match="unknown parameter"):
        get_strategy("ml_mlp", definitely_not_a_param=1)


@pytest.mark.parametrize("model, match", [
    ({"random_state": 3}, "managed by"),        # seeding is ours
    ({"early_stopping": True}, "managed by"),   # sklearn's split would be shuffled (leak)
    ({"solver": "lbfgs"}, "managed by"),        # partial_fit needs a stochastic solver
    ({"n_ensemble": 0}, ">= 1"),
    ({"max_epochs": 0}, ">= 1"),
    ({"not_an_mlp_option": 1}, "invalid MLPClassifier"),
])
def test_invalid_model_options_raise(model, match):
    with pytest.raises(ValueError, match=match):
        CalibratedMLP(model=model)


# ---------------------------------------------------------------------------------------
# point-in-time discipline and determinism
# ---------------------------------------------------------------------------------------
@pytest.fixture(scope="module")
def trend_md() -> MarketData:
    return MarketData(bars=make_synthetic_bars(3200, "H1", seed=21, model="trend",
                                               regime_params={"phi": 0.3}))


@pytest.fixture(scope="module")
def fitted_trend_mlp(trend_md) -> MLMLPStrategy:
    s = MLMLPStrategy(**COMMON, target="fixed_horizon", max_holding_bars=1)
    return s.fit(MarketData(bars=trend_md.bars.iloc[:2000]))


def test_ml_mlp_positive_control_trend(trend_md, fitted_trend_mlp):
    s = fitted_trend_mlp
    rep = s.fit_report_
    assert rep["skill_gate_passed"] and s.forecast_scalar_ > 0
    assert rep["n_ensemble"] == 2 and len(rep["best_epochs"]) == 2
    p = s.predict_proba(trend_md)
    lab = fixed_horizon_labels(trend_md.bars, 1)
    oos = lab[(lab["t_idx"] >= 2000) & (lab["label"] != 0)]
    auc = roc_auc_score(oos["label"] > 0, p.loc[oos.index])
    assert auc > 0.55, auc
    f = s.generate(trend_md)
    assert f.index.equals(trend_md.bars.index)
    assert f.between(-1, 1).all() and np.isfinite(f).all()
    assert (f.iloc[: s.warmup_bars] == 0).all()
    assert sharpe_tstat(f.iloc[2000:], trend_md.bars["close"].iloc[2000:]) > 2


def test_ml_mlp_leakage_future_perturbation(trend_md, fitted_trend_mlp):
    s = fitted_trend_mlp
    full = s.generate(trend_md)
    assert (full.iloc[2000:] != 0).mean() > 0.5
    for i, t in enumerate((2100, 2600, 3100)):
        alt = s.generate(perturb_future(trend_md, t, seed=i))
        pd.testing.assert_series_equal(alt.iloc[: t + 1], full.iloc[: t + 1], check_exact=True)
        assert not alt.iloc[t + 1:].equals(full.iloc[t + 1:])
        trunc = s.generate(MarketData(bars=trend_md.bars.iloc[: t + 1]))
        pd.testing.assert_series_equal(trunc, full.iloc[: t + 1], check_exact=True)


def test_ml_mlp_labels_end_inside_train_and_refit_is_deterministic(trend_md, fitted_trend_mlp):
    s = fitted_trend_mlp
    assert s.train_labels_["t1"].max() <= trend_md.bars.index[1999]
    again = MLMLPStrategy(**COMMON, target="fixed_horizon", max_holding_bars=1)
    again.fit(MarketData(bars=trend_md.bars.iloc[:2000]))
    pd.testing.assert_series_equal(again.generate(trend_md), s.generate(trend_md), check_exact=True)
    pd.testing.assert_series_equal(s.clone().generate(trend_md), s.generate(trend_md), check_exact=True)


def test_early_stopping_keeps_the_best_validation_epoch(trend_md):
    train = MarketData(bars=trend_md.bars.iloc[:2000])
    kw = dict(COMMON, target="fixed_horizon", max_holding_bars=1, importance="none", skill_gate_z=None)
    s = MLMLPStrategy(**{**kw, "model": {**FAST, "n_ensemble": 1, "max_epochs": 60, "patience": 5}})
    s.fit(train)
    rep = s.fit_report_
    best, run = rep["best_epochs"][0], rep["epochs_run"][0]
    assert 1 <= best <= run <= 60
    assert run == 60 or run - best == 5          # stopped by patience, or ran out of epochs
    off = MLMLPStrategy(**{**kw, "early_stopping": False,
                           "model": {**FAST, "n_ensemble": 1, "max_epochs": 7}})
    off.fit(train)
    assert off.fit_report_["best_epochs"] == [7] and off.fit_report_["epochs_run"] == [7]


# ---------------------------------------------------------------------------------------
# negative and positive controls
# ---------------------------------------------------------------------------------------
def test_ml_mlp_random_walk_has_no_edge():
    bars = make_synthetic_bars(5200, "H1", seed=33, model="gbm")
    md = MarketData(bars=bars)
    n_tr = 3200
    s = MLMLPStrategy(**COMMON, max_holding_bars=4, skill_gate_z=None, importance="none")
    s.fit(MarketData(bars=bars.iloc[:n_tr]))
    p = s.predict_proba(md)
    vol = ewm_vol(bars["close"]) * 2.0
    lab = triple_barrier_labels(bars, pt_mult=1, sl_mult=1, max_holding_bars=4, vol=vol,
                                vertical_label="sign")
    oos = lab[(lab["t_idx"] >= n_tr) & (lab["label"] != 0)]
    oos = oos[p.loc[oos.index].notna()]
    auc = roc_auc_score(oos["label"] > 0, p.loc[oos.index])
    assert 0.45 < auc < 0.55, auc
    f = s.generate(md)
    assert sharpe_tstat(f.iloc[n_tr:], bars["close"].iloc[n_tr:]) < 2.0
    gated = MLMLPStrategy(**COMMON, max_holding_bars=4, importance="none")
    gated.fit(MarketData(bars=bars.iloc[:n_tr]))
    assert not gated.fit_report_["skill_gate_passed"]
    assert gated.forecast_scalar_ == 0 and (gated.generate(md) == 0).all()


def test_ml_mlp_planted_feature_is_found_and_explained():
    md, feats = planted_market(3000, seed=1)
    n_tr = 2000
    s = MLMLPStrategy(**{**COMMON, "importance_repeats": 3}, max_holding_bars=12)
    s.fit(MarketData(bars=md.bars.iloc[:n_tr]), features=feats)
    assert s.fit_report_["skill_gate_passed"]
    assert s.feature_importances_.index[0] == "planted"
    p = s.predict_proba(md, features=feats)
    vol = ewm_vol(md.bars["close"]) * math.sqrt(12)
    lab = triple_barrier_labels(md.bars, pt_mult=1, sl_mult=1, max_holding_bars=12, vol=vol,
                                vertical_label="sign")
    oos = lab[(lab["t_idx"] >= n_tr) & (lab["label"] != 0)]
    assert roc_auc_score(oos["label"] > 0, p.loc[oos.index]) > 0.55
    f = s.generate(md, features=feats)
    assert np.corrcoef(f.iloc[n_tr:], feats["planted"].iloc[n_tr:])[0, 1] > 0.5
    info = s.explain()
    json.dumps(info)
    assert info["model"]["hidden_layer_sizes"] == [16]
    assert "planted" in info["top_features"]
