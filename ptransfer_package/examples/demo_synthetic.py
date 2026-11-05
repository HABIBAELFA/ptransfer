"""Worked example on a synthetic cohort: should a centre adapt an imported model?

Runs in about a minute on a laptop. Replace the synthetic arrays with your own
transitions (X: inputs at visit j, D: change to visit j+1, subject: patient id).
"""
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

import ptransfer as pt

rng = np.random.default_rng(0)


def cohort(n_pat, shift, seed):
    r = np.random.default_rng(seed)
    subject = np.repeat(np.arange(n_pat), r.integers(1, 4, n_pat))
    X = r.standard_normal((len(subject), 3)) + shift
    lam = np.array([0.6, 0.35, 0.12])
    D = -lam * (X - shift) + 0.3 * np.tanh(X) + 0.6 * r.standard_normal((len(subject), 3))
    return X, D, subject


# 1. The provider trains a model on its (large) source cohort -----------------
Xs, Ds, ss = cohort(8000, 0.0, 1)
gbm = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4)
source_models = [gbm.__class__(**gbm.get_params()).fit(Xs, Ds[:, j]) for j in range(3)]
source_predict = lambda X: np.column_stack([m.predict(X) for m in source_models])

# Before any local data exist: which outputs will need the most local patients?
print('source R2 by output:', np.round(pt.source_r2(Xs, Ds), 3))

# 2. The adopting centre holds a pool of local transitions and a test set -----
Xp, Dp, sp = cohort(1500, 0.3, 2)
Xt, Dt, st = cohort(800, 0.3, 3)

strategies = {
    'frozen': lambda: pt.Frozen(source_predict),
    'recalibration, shrunk': lambda: pt.Recalibration(source_predict, loss='abs', shrink=True),
    'local': lambda: pt.LocalModel(gbm, shrink=False),
    'residual boosting, shrunk': lambda: pt.ResidualBoosting(source_predict, gbm, shrink=True),
}
res = pt.compare_strategies(strategies, Xp, Dp, sp, Xt, Dt, st,
                            sizes=[20, 50, 100, 300, 1000], n_draws=5, seed=0,
                            output_names=['A', 'B', 'C'])

# 3. Expected performance with 95% CIs, and a plain-language recommendation --
table = pt.bootstrap_table(res, n_boot=1000)
print(table.round(4).to_string(index=False))
print(pt.recommend(table))
print('local model beats persistence from:', pt.threshold_with_ci(res, 'local', n_boot=500))
