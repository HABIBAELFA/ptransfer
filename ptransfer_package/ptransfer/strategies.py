"""Transfer strategies for an imported model of change.

Every strategy exposes fit(X, D, subject) and predict(X) -> (n, k) predicted
change in physical units. The imported model is passed as a callable
`source_predict(X) -> (n, k)`; its training data are never needed, except by
Pooled, which is included for comparison only.
"""
import numpy as np
from sklearn.base import clone
from sklearn.model_selection import GroupKFold

DEFAULT_GRID = (0.0, 0.1, 0.2, 0.35, 0.5, 0.7, 0.85, 1.0)


def cv_shrink_weight(fit_predict, X, y, base, groups, grid=DEFAULT_GRID, n_splits=5):
    """Choose w in `grid` minimising absolute error of base + w * (fit - base)
    by cross-validation grouped by patient, inside the local cohort only.
    Returns 0 (the safe default) when there are too few patients."""
    groups = np.asarray(groups)
    n_grp = len(np.unique(groups))
    if len(y) < 10 or n_grp < 3:
        return 0.0
    scores = np.zeros(len(grid))
    for tr, va in GroupKFold(n_splits=min(n_splits, n_grp)).split(X, y, groups):
        f = fit_predict(X[tr], y[tr], X[va])
        for j, w in enumerate(grid):
            scores[j] += np.abs(y[va] - (base[va] + w * (f - base[va]))).sum()
    return float(grid[int(np.argmin(scores))])


class Frozen:
    """The imported model applied unchanged. Uses no local data."""
    def __init__(self, source_predict):
        self.source_predict = source_predict

    def fit(self, X, D, subject):
        return self

    def predict(self, X):
        return np.asarray(self.source_predict(X), float)


class Recalibration:
    """One slope per output multiplying the imported prediction.
    loss='abs' fits the slope by absolute error (weighted median), matching the
    evaluation metric; loss='ls' by least squares. shrink=True moves the slope
    toward 1 by grouped cross-validation."""
    def __init__(self, source_predict, loss='abs', shrink=True, grid=DEFAULT_GRID):
        self.source_predict, self.loss, self.shrink, self.grid = source_predict, loss, shrink, grid

    def _slope(self, p, d):
        ok = np.abs(p) > 1e-12
        if not ok.any():
            return 1.0
        if self.loss == 'ls':
            return float((p[ok] * d[ok]).sum() / (p[ok] ** 2).sum())
        q, w = d[ok] / p[ok], np.abs(p[ok])
        o = np.argsort(q)
        cum = np.cumsum(w[o])
        return float(q[o][np.searchsorted(cum, 0.5 * cum[-1])])

    def fit(self, X, D, subject):
        P = np.asarray(self.source_predict(X), float)
        D = np.asarray(D, float)
        self.c_ = np.ones(D.shape[1])
        for j in range(D.shape[1]):
            c = self._slope(P[:, j], D[:, j])
            if self.shrink:
                Pj = P[:, j]
                # refit the slope on the training fold (indices a), predict the validation fold (b)
                fp = lambda a, ya, b, Pj=Pj: self._slope(Pj[a], ya) * Pj[b]
                w = cv_shrink_weight(fp, np.arange(len(D)), D[:, j], Pj, subject, self.grid)
                c = 1.0 + w * (c - 1.0)
            self.c_[j] = c
        return self

    def predict(self, X):
        return np.asarray(self.source_predict(X), float) * self.c_


class LocalModel:
    """A model trained on the local cohort alone, one estimator per output.
    shrink=True multiplies its prediction by a cross-validated weight, i.e.
    shrinks it toward persistence (zero change)."""
    def __init__(self, estimator, shrink=True, grid=DEFAULT_GRID):
        self.estimator, self.shrink, self.grid = estimator, shrink, grid

    def fit(self, X, D, subject):
        X, D = np.asarray(X, float), np.asarray(D, float)
        self.models_, self.w_, self.s_ = [], np.ones(D.shape[1]), D.std(axis=0, ddof=1)
        self.s_[~np.isfinite(self.s_) | (self.s_ <= 0)] = 1.0
        for j in range(D.shape[1]):
            y = D[:, j] / self.s_[j]
            m = clone(self.estimator).fit(X, y)
            self.models_.append(m)
            if self.shrink:
                fp = lambda a, ya, b: clone(self.estimator).fit(X[a], ya).predict(X[b])
                self.w_[j] = cv_shrink_weight(fp, np.arange(len(y)), y, np.zeros(len(y)), subject, self.grid)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        return np.column_stack([m.predict(X) * s * w for m, s, w in zip(self.models_, self.s_, self.w_)])


class ResidualBoosting:
    """Continues the imported model on local data: a model of the residual
    D - source_predict(X) is added to the imported prediction. shrink=True
    (recommended) scales the correction by a cross-validated weight, i.e.
    shrinks it toward the imported model."""
    def __init__(self, source_predict, estimator, shrink=True, grid=DEFAULT_GRID):
        self.source_predict, self.estimator, self.shrink, self.grid = source_predict, estimator, shrink, grid

    def fit(self, X, D, subject):
        X, D = np.asarray(X, float), np.asarray(D, float)
        R = D - np.asarray(self.source_predict(X), float)
        self.models_, self.w_, self.s_ = [], np.ones(D.shape[1]), D.std(axis=0, ddof=1)
        self.s_[~np.isfinite(self.s_) | (self.s_ <= 0)] = 1.0
        for j in range(D.shape[1]):
            y = R[:, j] / self.s_[j]
            self.models_.append(clone(self.estimator).fit(X, y))
            if self.shrink:
                fp = lambda a, ya, b: clone(self.estimator).fit(X[a], ya).predict(X[b])
                self.w_[j] = cv_shrink_weight(fp, np.arange(len(y)), y, np.zeros(len(y)), subject, self.grid)
        return self

    def predict(self, X):
        X = np.asarray(X, float)
        corr = np.column_stack([m.predict(X) * s * w for m, s, w in zip(self.models_, self.s_, self.w_)])
        return np.asarray(self.source_predict(X), float) + corr
