import numpy as np
import pytest
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

import ptransfer as pt


def synth(n_pat=400, k=3, lam=(0.4, 0.2, 0.05), seed=0, shift=0.0):
    rng = np.random.default_rng(seed)
    subj = np.repeat(np.arange(n_pat), rng.integers(1, 4, n_pat))
    x = rng.standard_normal((len(subj), k)) + shift
    D = -np.asarray(lam) * (x - shift) + rng.standard_normal((len(subj), k))
    return x, D, subj


def test_persistence_is_exactly_one():
    x, D, s = synth()
    r = pt.relative_error(D, np.zeros_like(D), s)
    assert r['composite'] == pytest.approx(1.0)
    assert np.allclose(r['per_output'], 1.0)


def test_perfect_prediction_is_zero():
    x, D, s = synth()
    assert pt.relative_error(D, D, s)['composite'] == pytest.approx(0.0)


def test_within_patient_averaging():
    D = np.array([[1.0], [1.0], [1.0], [4.0]])
    E, ids = pt.patient_errors(D, np.zeros_like(D), np.array([0, 0, 0, 1]))
    assert np.allclose(E[:, 0], [1.0, 4.0])      # patient 0 counts once, not three times


def test_circular_wrapping():
    D = np.array([[179.0], [-179.0]])
    P = np.array([[-179.0], [179.0]])
    E, _ = pt.patient_errors(D, P, np.array([0, 1]), circular=[True])
    assert np.allclose(E[:, 0], 2.0)


def test_durable_threshold():
    assert pt.durable_threshold([10, 20, 40, 80], [1.1, 0.99, 1.01, 0.95]) == 80
    assert pt.durable_threshold([10, 20, 40], [0.9, 0.9, 0.9]) == 10
    assert np.isnan(pt.durable_threshold([10, 20], [1.2, 1.1]))


def test_source_r2_orders_signal():
    x, D, s = synth(n_pat=3000)
    r2 = pt.source_r2(x, D)
    assert r2[0] > r2[1] > r2[2]


def test_shrink_weight_zero_without_signal():
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 2)); y = rng.standard_normal(200); g = np.arange(200) // 2
    fp = lambda a, ya, b: LinearRegression().fit(X[a], ya).predict(X[b])
    assert pt.cv_shrink_weight(fp, np.arange(200), y, np.zeros(200), g) <= 0.2


def test_shrink_weight_one_with_strong_signal():
    rng = np.random.default_rng(2)
    X = rng.standard_normal((300, 1)); y = 3 * X[:, 0] + 0.1 * rng.standard_normal(300); g = np.arange(300)
    fp = lambda a, ya, b: LinearRegression().fit(X[a], ya).predict(X[b])
    assert pt.cv_shrink_weight(fp, np.arange(300), y, np.zeros(300), g) == 1.0


def test_recalibration_mae_slope_is_weighted_median():
    p = np.array([1.0, 1.0, 1.0]); d = np.array([0.5, 2.0, 3.0])
    r = pt.Recalibration(lambda X: None, loss='abs', shrink=False)
    assert r._slope(p, d) == 2.0


def test_end_to_end_frozen_beats_small_local():
    xs, Ds, ss = synth(n_pat=3000, seed=3)
    src = [LinearRegression().fit(xs, Ds[:, j]) for j in range(Ds.shape[1])]
    source_predict = lambda X: np.column_stack([m.predict(X) for m in src])
    xp, Dp, sp = synth(n_pat=600, seed=4)
    xt, Dt, st = synth(n_pat=400, seed=5)
    gbm = HistGradientBoostingRegressor(max_iter=50, max_depth=3)
    strategies = {
        'frozen': lambda: pt.Frozen(source_predict),
        'local': lambda: pt.LocalModel(gbm, shrink=False),
        'residual': lambda: pt.ResidualBoosting(source_predict, gbm, shrink=False),
        'residual, shrunk': lambda: pt.ResidualBoosting(source_predict, gbm, shrink=True),
    }
    res = pt.compare_strategies(strategies, xp, Dp, sp, xt, Dt, st, sizes=[20, 300], n_draws=3, seed=0)
    tab = pt.bootstrap_table(res, n_boot=200)
    small_local = tab[(tab.n == 20) & (tab.strategy == 'local')].iloc[0]
    assert small_local.verdict == 'reference better'
    d = tab.set_index(['strategy', 'n'])['diff']
    assert d[('residual, shrunk', 20)] <= d[('residual', 20)] + 1e-3   # shrinkage never costs more
    assert abs(d[('residual, shrunk', 300)]) < 0.01
    th = pt.threshold_with_ci(res, 'local', n_boot=100)
    assert set(th) == {'threshold', 'lo', 'hi', 'never'}
    rec = pt.recommend(tab)
    assert 'local' in rec
