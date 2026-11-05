"""Parametric simulation: when does adapting an imported model help?

Data-generating process (one output, three inputs, change between visits):
    source : x ~ N(0, I3);          d = f(x1) + u_i + e
    target : x ~ N(delta/2, I3);     d = sqrt(1-delta) f(x1) + sqrt(delta) f(x2) + c*delta + u_i + e
             (same signal variance; a share delta of it moves to x2, which the source ignores)
    f(x)   = -lam * (x + 0.5 tanh(2x))          (nonlinear mean reversion)
    u_i    : patient random effect, sd 0.3 * sd(e)
    e      : Gaussian, or Student t(3) scaled to the same variance
The noise variance is set so that the source R^2 of the signal equals R2.
x2, x3 carry no signal (as the other ECG measurements do for a given output).

Parameters varied (one factor at a time around a base case, plus a full
R2 x delta grid):
    R2      in {0.05, 0.10, 0.20, 0.30}      signal strength
    delta   in {0, 0.10, 0.25, 0.50}         share of the signal that moves (domain shift)
    noise   in {gaussian, t3}
    visits  in {1, 3}                        mean transitions per patient
    learner in {boosting, linear}
Everything else is fixed: source 20,000 patients, target test 4,000 patients,
target reservoir 6,000 patients; sizes 20..5,000.
"""
import json
import sys
import time
from itertools import product

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import LinearRegression

import ptransfer as pt

SIZES = [20, 50, 100, 200, 500, 1000, 2000, 5000]
DRAWS = {20: 10, 50: 10, 100: 10, 200: 10, 500: 6, 1000: 6, 2000: 4, 5000: 3}
N_SRC, N_TEST, N_POOL = 20000, 4000, 6000


def gbm():
    return HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_depth=4,
                                         early_stopping=True, validation_fraction=0.2,
                                         n_iter_no_change=15, random_state=0)


def f(x, lam):
    return -lam * (x + 0.5 * np.tanh(2 * x))


def noise_sd(R2, lam):
    z = np.random.default_rng(99).standard_normal(200000)
    v = f(z, lam).var()
    return np.sqrt(v * (1 - R2) / R2)


def cohort(n_pat, R2, delta, noise, visits, seed, target):
    rng = np.random.default_rng(seed)
    lam = 0.5
    sd_e = noise_sd(R2, lam)
    reps = np.ones(n_pat, int) if visits == 1 else 1 + rng.poisson(visits - 1, n_pat)
    subj = np.repeat(np.arange(n_pat), reps)
    n = len(subj)
    shift = delta / 2 if target else 0.0
    X = rng.standard_normal((n, 3)) + shift
    if noise == 'gaussian':
        e = rng.standard_normal(n)
    else:
        e = rng.standard_t(3, n) / np.sqrt(3.0)
    e *= sd_e / np.sqrt(1 + 0.09)
    u = (0.3 * sd_e / np.sqrt(1 + 0.09)) * rng.standard_normal(n_pat)[subj]
    sig = f(X[:, 0] - shift, lam)
    if target:
        # part of the signal moves to an input the source model ignores (x2), plus a drift:
        # a recalibration slope cannot repair this, a local model of the residual can
        sd_f = np.sqrt(f(np.random.default_rng(1).standard_normal(10000), lam).var())
        sig = np.sqrt(1 - delta) * sig + np.sqrt(delta) * f(X[:, 1] - shift, lam) + 0.3 * delta * sd_f
    D = (sig + u + e)[:, None]
    return X, D, subj


def run(scn, seed=0):
    t0 = time.time()
    R2, delta, noise, visits, learner = scn['R2'], scn['delta'], scn['noise'], scn['visits'], scn['learner']
    est = gbm() if learner == 'boosting' else LinearRegression()
    Xs, Ds, ss = cohort(N_SRC, R2, delta, noise, visits, seed + 1, target=False)
    Xt, Dt, st = cohort(N_TEST, R2, delta, noise, visits, seed + 2, target=True)
    Xp, Dp, sp = cohort(N_POOL, R2, delta, noise, visits, seed + 3, target=True)
    src = (gbm() if learner == 'boosting' else LinearRegression()).fit(Xs, Ds[:, 0])
    source_predict = lambda X: src.predict(X)[:, None]
    strategies = {
        'frozen': lambda: pt.Frozen(source_predict),
        'local': lambda: pt.LocalModel(est, shrink=False),
        'local, shrunk': lambda: pt.LocalModel(est, shrink=True),
        'recalibration, shrunk': lambda: pt.Recalibration(source_predict, loss='abs', shrink=True),
        'residual, shrunk': lambda: pt.ResidualBoosting(source_predict, est, shrink=True),
    }
    res = pt.compare_strategies(strategies, Xp, Dp, sp, Xt, Dt, st, SIZES, DRAWS, seed=seed)
    tab = pt.bootstrap_table(res, n_boot=1000, seed=seed)
    frozen = float(res.ref_errors.mean() / res.pers_errors.mean())
    out = dict(scn, frozen=frozen, source_r2=float(pt.source_r2(Xs, Ds)[0]), table=tab.to_dict('records'))
    for s in ['local', 'local, shrunk']:
        out[f'thr_{s}'] = pt.threshold_with_ci(res, s, n_boot=500, seed=seed)
    # benefit threshold: smallest size from which the strategy is better than frozen at every larger size
    for s in ['residual, shrunk', 'recalibration, shrunk', 'local']:
        g = tab[tab.strategy == s].sort_values('n')
        better = (g.verdict == 'strategy better').to_numpy()
        ben = next((int(n) for i, n in enumerate(g.n) if better[i:].all()), None)
        worse = g[g.verdict == 'reference better']
        out[f'benefit_{s}'] = ben
        out[f'maxcost_{s}'] = float(g['diff'].max())
        out[f'worse_upto_{s}'] = int(worse.n.max()) if len(worse) else None
    out['seconds'] = time.time() - t0
    return out


def scenarios():
    base = dict(R2=0.20, delta=0.25, noise='gaussian', visits=2, learner='boosting')
    grid = [dict(base, R2=r, delta=d) for r, d in product([0.05, 0.10, 0.20, 0.30], [0.0, 0.10, 0.25, 0.5])]
    extra = [dict(base, noise='t3'), dict(base, visits=1), dict(base, visits=3), dict(base, learner='linear')]
    return grid + extra


if __name__ == '__main__':
    idx = [int(a) for a in sys.argv[1:]] or list(range(len(scenarios())))
    for i in idx:
      for rep in range(3):
        s = scenarios()[i]
        r = run(s, seed=1000 * rep + 100 + i)
        r['rep'] = rep
        json.dump(r, open(f'res_{i:02d}_{rep}.json', 'w'), default=float)
        print(i, rep, s, 'frozen', round(r['frozen'], 4), 'thr local', r['thr_local']['threshold'],
              'benefit resid', r['benefit_residual, shrunk'], f"{r['seconds']:.0f}s", flush=True)
