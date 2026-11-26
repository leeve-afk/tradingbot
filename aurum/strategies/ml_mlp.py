"""``ml_mlp`` - a small feed-forward neural network in place of ``ml_gbm``'s boosted trees.

Everything except the classifier is ``ml_gbm`` (:class:`aurum.strategies.ml.MLGBMStrategy`):
the same causal FeaturePipeline features (robust-scaled and clipped to +/-5), the same
triple-barrier labels purged at the end of the training slice, the same uniqueness weights,
class balancing, time-ordered validation tail, Platt calibration, dead-zone, Carver scaling
and skill gate. Only :class:`CalibratedMLP` replaces :class:`~aurum.strategies.ml.CalibratedGBM`,
so a difference between the two strategies' walk-forward results is a difference between
the model families, not between two pipelines.

The network (Gu, Kelly & Xiu 2020, "Empirical Asset Pricing via Machine Learning", RFS
33(5): shallow ReLU networks, a geometric pyramid of hidden widths, L2 penalty, Adam, early
stopping and an ensemble over random seeds):

* ``hidden_layer_sizes`` (32, 16), ReLU, L2 penalty ``alpha``, Adam (Kingma & Ba 2015),
  mini-batches of ``batch_size``;
* trained one epoch at a time with ``partial_fit`` on a fresh, seeded permutation of the fit
  set; after each epoch the weighted log-loss on the purged, time-ordered validation tail
  is recorded and the best epoch's weights are kept (``patience`` epochs without an
  improvement stop training). sklearn's own ``early_stopping`` is NOT used: it would carve a
  *shuffled* validation split out of overlapping labels, i.e. leak;
* ``n_ensemble`` members with seeds ``seed, seed+1, ...``; their raw probabilities are
  averaged before calibration (seed averaging reduces the variance of a network fitted to
  a low signal-to-noise target).

Permutation importance is computed by hand (weighted validation-AUC drop per permuted
column, Breiman 2001), like ``ml_gbm``'s, so ``explain()`` and ``importance_by_group()``
work unchanged.
"""

from __future__ import annotations

import logging
import math
import warnings
from contextlib import nullcontext
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from aurum.strategies.base import register_strategy
from aurum.strategies.ml import CalibratedGBM, MLGBMStrategy, _weighted_auc, _weighted_logloss

logger = logging.getLogger(__name__)

__all__ = ["CalibratedMLP", "MLMLPStrategy"]


class _MLPEnsemble:
    """Average of the members' ``P(y = 1)``; exposes ``predict_proba`` like an sklearn model."""

    def __init__(self, members: list[Any]) -> None:
        self.members = members

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = np.mean([m.predict_proba(X)[:, 1] for m in self.members], axis=0)
        return np.column_stack([1.0 - p, p])


class CalibratedMLP(CalibratedGBM):
    """``MLPClassifier`` ensemble + time-ordered validation tail + calibration.

    Same contract as :class:`~aurum.strategies.ml.CalibratedGBM` (fit set / purged
    validation tail split, separate class balancing, calibration, report keys,
    permutation importance); only the model and its early stopping differ (see module doc).
    """

    DEFAULT_MODEL: ClassVar[dict[str, Any]] = {
        "hidden_layer_sizes": (32, 16),
        "activation": "relu",
        "alpha": 1e-3,
        "learning_rate_init": 1e-3,
        "batch_size": 256,
        "max_epochs": 100,
        "patience": 10,
        "n_ensemble": 3,
    }
    RESERVED_MODEL_KEYS: ClassVar[frozenset[str]] = frozenset(
        {"class_weight", "early_stopping", "random_state", "validation_fraction", "n_iter_no_change",
         "max_iter", "warm_start", "shuffle", "solver"})
    #: keys consumed here rather than passed to ``MLPClassifier``
    _OWN_KEYS: ClassVar[tuple[str, ...]] = ("max_epochs", "patience", "n_ensemble")

    def __init__(self, **kwargs: Any) -> None:
        from sklearn.neural_network import MLPClassifier

        super().__init__(**kwargs)
        mp = self.model_params
        if mp.get("hidden_layer_sizes") is not None:
            mp["hidden_layer_sizes"] = tuple(int(h) for h in np.atleast_1d(mp["hidden_layer_sizes"]))
        for key in self._OWN_KEYS:
            if int(mp[key]) < 1:
                raise ValueError(f"model param {key} must be >= 1, got {mp[key]!r}")
        try:
            MLPClassifier(**self._mlp_kwargs(seed=0))
        except TypeError as exc:
            raise ValueError(f"invalid MLPClassifier params: {exc}") from exc

    def _mlp_kwargs(self, seed: int) -> dict[str, Any]:
        kw = {k: v for k, v in self.model_params.items() if k not in self._OWN_KEYS}
        return {**kw, "solver": "adam", "shuffle": False, "early_stopping": False,
                "random_state": int(seed), "max_iter": 1}

    # ---- threading: the MLP's matrix products run in BLAS, not OpenMP --------------------
    def _threads(self):
        if self.n_threads is None:
            return nullcontext()
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=int(self.n_threads))

    # ---- fit ------------------------------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: np.ndarray, sample_weight: np.ndarray, t_idx: np.ndarray,
            t1_idx: np.ndarray) -> CalibratedMLP:
        # Split, purge and weighting mirror CalibratedGBM.fit line for line (see its comments).
        y = np.asarray(y, dtype=float)
        w = np.asarray(sample_weight, dtype=float)
        t0 = np.asarray(t_idx, dtype=np.int64)
        t1 = np.asarray(t1_idx, dtype=np.int64)
        if not (len(X) == len(y) == len(w) == len(t0) == len(t1)):
            raise ValueError("X, y, sample_weight, t_idx, t1_idx must have equal length")
        if np.any(np.diff(t0) < 0):
            raise ValueError("events must be in chronological order (sorted t_idx)")
        n = len(y)
        n_val = int(round(self.val_frac * n))
        if n - n_val < self.min_events or n_val < max(20, self.min_events // 10):
            raise ValueError(f"not enough training events ({n}; need >= {self.min_events} to fit "
                             f"and a validation tail)")
        pos = np.arange(n)
        val_mask = pos >= n - n_val
        val_start = int(t0[n - n_val])
        fit_mask = ~val_mask & (t1 < val_start - self.embargo_bars)
        n_purged = int((~val_mask).sum() - fit_mask.sum())
        y_fit, y_val = y[fit_mask], y[val_mask]
        if len(np.unique(y_fit)) < 2 or len(np.unique(y_val)) < 2:
            raise ValueError("both classes must be present in the fit set and the validation tail")
        if not (np.all(np.isfinite(w)) and np.all(w >= 0) and w[fit_mask].sum() > 0
                and w[val_mask].sum() > 0):
            raise ValueError("sample weights must be finite, >= 0 and have a positive sum in both "
                             "the fit set and the validation tail (is time_decay zeroing the fit set?)")
        cw = np.ones(n)
        if self.class_weight == "balanced":
            for part in (fit_mask, val_mask):
                wp, yp = w[part], y[part]
                tot = wp.sum()
                for cls in (0.0, 1.0):
                    share = wp[yp == cls].sum() / tot
                    cw[part & (y == cls)] = 0.5 / max(share, 1e-12)
        w_all = w * cw
        Xa = np.ascontiguousarray(X.to_numpy(dtype=float))
        if not np.all(np.isfinite(Xa)):
            raise ValueError("CalibratedMLP needs finite features (no NaN/inf)")
        X_fit, X_val = Xa[fit_mask], Xa[val_mask]
        w_fit, w_val = w_all[fit_mask], w_all[val_mask]
        w_fit = w_fit / w_fit.mean()
        w_val = w_val / w_val.mean()
        self.feature_names_ = [str(c) for c in X.columns]

        base_rate = float(np.sum(w_val * y_val) / np.sum(w_val))
        base_loss = _weighted_logloss(y_val, np.full(len(y_val), base_rate), w_val)
        n_members = int(self.model_params["n_ensemble"])
        members, best_epochs, curves = [], [], []
        with self._threads(), warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)  # overflow in exp of saturated units
            for k in range(n_members):
                net, best_epoch, losses = self._train_member(X_fit, y_fit, w_fit, X_val, y_val, w_val,
                                                             seed=self.seed + k)
                members.append(net)
                best_epochs.append(best_epoch)
                curves.append(losses)
            model = _MLPEnsemble(members)
            p_val_raw = model.predict_proba(X_val)[:, 1]
            p_fit_raw = model.predict_proba(X_fit)[:, 1]
        self.model_ = model
        self.calibrator_ = self._fit_calibrator(p_val_raw, y_val, w_val)
        p_val = self._calibrate(p_val_raw)
        self._val = (X_val, y_val, w_val)
        max_epochs = int(self.model_params["max_epochs"])
        self.report_ = {
            "n_events": n,
            "n_fit": int(fit_mask.sum()),
            "n_val": int(n_val),
            "n_purged_before_val": n_purged,
            "fit_pos_rate": float(y_fit.mean()),
            "val_pos_rate": float(y_val.mean()),
            "best_iter": int(round(float(np.mean(best_epochs)))),
            "best_epochs": [int(e) for e in best_epochs],
            "epochs_run": [len(c) for c in curves],
            "max_iter": max_epochs,
            "n_ensemble": n_members,
            "fit_auc": _weighted_auc(y_fit, p_fit_raw, w_fit),
            "val_auc": _weighted_auc(y_val, p_val_raw, w_val),
            "val_logloss_base": base_loss,
            "val_logloss_model": _weighted_logloss(y_val, p_val_raw, w_val),
            "val_logloss_calibrated": _weighted_logloss(y_val, p_val, w_val),
            "val_p_mean": float(np.mean(p_val)),
            "val_p_std": float(np.std(p_val)),
            "calibration": self.calibrator_.get("kind"),
            "calibration_params": {k: v for k, v in self.calibrator_.items()
                                   if k in ("a", "b", "clamped")},
            "n_features": X.shape[1],
        }
        logger.info("CalibratedMLP: %d fit / %d val events, %d members, best epochs=%s, val AUC=%.4f, "
                    "logloss base=%.5f model=%.5f cal=%.5f", self.report_["n_fit"], n_val, n_members,
                    best_epochs, self.report_["val_auc"], base_loss, self.report_["val_logloss_model"],
                    self.report_["val_logloss_calibrated"])
        return self

    def _train_member(self, X_fit: np.ndarray, y_fit: np.ndarray, w_fit: np.ndarray,
                      X_val: np.ndarray, y_val: np.ndarray, w_val: np.ndarray, *, seed: int
                      ) -> tuple[Any, int, list[float]]:
        """One network, trained epoch by epoch; keeps the weights of the epoch with the lowest
        weighted validation log-loss (the last epoch when ``early_stopping`` is off)."""
        from sklearn.neural_network import MLPClassifier

        net = MLPClassifier(**self._mlp_kwargs(seed))
        rng = np.random.default_rng(seed)
        classes = np.array([0.0, 1.0])
        max_epochs = int(self.model_params["max_epochs"])
        patience = int(self.model_params["patience"])
        losses: list[float] = []
        best_loss, best_epoch, best_state = math.inf, 0, None
        for epoch in range(1, max_epochs + 1):
            perm = rng.permutation(len(y_fit))
            net.partial_fit(X_fit[perm], y_fit[perm], sample_weight=w_fit[perm], classes=classes)
            loss = _weighted_logloss(y_val, net.predict_proba(X_val)[:, 1], w_val)
            losses.append(loss)
            if not self.early_stopping:
                continue
            if loss < best_loss:
                best_loss, best_epoch = loss, epoch
                best_state = ([c.copy() for c in net.coefs_], [b.copy() for b in net.intercepts_])
            elif epoch - best_epoch >= patience:
                break
        if self.early_stopping and best_state is not None:
            net.coefs_, net.intercepts_ = best_state
        else:
            best_epoch = len(losses)
        return net, best_epoch, losses

    # ---- importance -------------------------------------------------------------------------
    def compute_permutation_importance(self, *, n_repeats: int = 3, max_rows: int | None = 5000,
                                       seed: int | None = None) -> pd.Series:
        """Mean drop in weighted validation AUC when each feature is permuted (Breiman 2001)."""
        if self._val is None:
            raise RuntimeError("validation tail not available (released or not fitted)")
        X_val, y_val, w_val = self._val
        rng = np.random.default_rng(self.seed if seed is None else seed)
        if max_rows is not None and len(y_val) > max_rows:
            sel = np.sort(rng.choice(len(y_val), size=int(max_rows), replace=False))
            X_val, y_val, w_val = X_val[sel], y_val[sel], w_val[sel]
        with self._threads():
            base = _weighted_auc(y_val, self.model_.predict_proba(X_val)[:, 1], w_val)
            drops = np.zeros((X_val.shape[1], int(n_repeats)))
            for j in range(X_val.shape[1]):
                Xp = X_val.copy()
                for r in range(int(n_repeats)):
                    Xp[:, j] = X_val[rng.permutation(len(y_val)), j]
                    drops[j, r] = base - _weighted_auc(y_val, self.model_.predict_proba(Xp)[:, 1], w_val)
        imp = pd.Series(drops.mean(axis=1), index=self.feature_names_, name="importance")
        std = pd.Series(drops.std(axis=1), index=self.feature_names_, name="importance_std")
        order = imp.sort_values(ascending=False).index
        self.feature_importances_ = imp.loc[order]
        self.feature_importances_std_ = std.loc[order]
        return self.feature_importances_


@register_strategy
class MLMLPStrategy(MLGBMStrategy):
    """Neural-network direction classifier: ``ml_gbm`` with an MLP ensemble as the model.

    Rationale: boosted trees combine weak conditional effects through axis-aligned splits;
    a small ReLU network combines them through smooth interactions of all features at once,
    which suits signals that are linear-ish combinations of many correlated inputs (Gu,
    Kelly & Xiu (2020): shallow networks were the strongest return predictors among the
    model families they compared). Everything that guards against overfitting - purged
    triple-barrier labels, uniqueness weights, a time-ordered validation tail for early
    stopping and calibration, and the skill gate - is inherited from ``ml_gbm`` unchanged,
    so the two strategies differ only in the model family. Model options go in ``model``
    (see :attr:`CalibratedMLP.DEFAULT_MODEL`).
    """

    name = "ml_mlp"
    description = ("Small ReLU MLP ensemble (sklearn MLPClassifier, Adam, L2, epoch-wise early "
                   "stopping on a purged time-ordered validation tail) on ml_gbm's causal features "
                   "and triple-barrier labels; calibrated 2p-1 with dead-zone, Carver scaling and "
                   "the same skill gate.")

    def _new_classifier(self, class_weight: str | None) -> CalibratedMLP:
        return CalibratedMLP(
            model=self.params["model"],
            val_frac=float(self.params["val_frac"]),
            embargo_bars=int(self.params["embargo_bars"]),
            early_stopping=bool(self.params["early_stopping"]),
            calibration=self.params["calibration"],
            class_weight=class_weight,
            min_events=int(self.params["min_train_events"]),
            seed=int(self.params["seed"]),
            n_threads=self.params["n_threads"],
        )
