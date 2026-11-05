"""Compare transfer strategies across local cohort sizes, with a two-level
bootstrap over (i) which local patients a centre happens to have and
(ii) the patients of the test set."""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .core import default_scale, durable_threshold, patient_errors


@dataclass
class Comparison:
    sizes: list
    reference: str
    errors: dict = field(default_factory=dict)      # errors[n][name] -> (R, P, k)
    ref_errors: np.ndarray = None                    # (P, k)
    pers_errors: np.ndarray = None                   # (P, k)
    output_names: list = None


def compare_strategies(strategies, X_pool, D_pool, subj_pool, X_test, D_test, subj_test,
                       sizes, n_draws, reference='frozen', circular=None, seed=0,
                       output_names=None):
    """Fit every strategy on the same random draws of n local patients.

    strategies : dict name -> zero-argument factory returning a fresh strategy
                 (must include `reference`, typically the frozen imported model)
    n_draws    : int, or dict size -> number of draws
    Returns a Comparison holding per-patient test errors for every draw."""
    rng = np.random.default_rng(seed)
    scale = default_scale(D_test, circular)
    pers, ids = patient_errors(D_test, np.zeros_like(np.asarray(D_test, float)), subj_test, circular, scale)
    ref = strategies[reference]().fit(X_pool[:0], D_pool[:0], subj_pool[:0])
    ref_err, _ = patient_errors(D_test, ref.predict(X_test), subj_test, circular, scale)
    pool_ids = np.unique(subj_pool)
    res = Comparison(sizes=[n for n in sizes if n <= len(pool_ids)], reference=reference,
                     ref_errors=ref_err, pers_errors=pers, output_names=output_names)
    for n in res.sizes:
        reps = n_draws[n] if isinstance(n_draws, dict) else n_draws
        acc = {name: [] for name in strategies if name != reference}
        for _ in range(reps):
            m = np.isin(subj_pool, rng.choice(pool_ids, size=n, replace=False))
            for name in acc:
                s = strategies[name]().fit(X_pool[m], D_pool[m], subj_pool[m])
                e, _ = patient_errors(D_test, s.predict(X_test), subj_test, circular, scale)
                acc[name].append(e)
        res.errors[n] = {k: np.stack(v) for k, v in acc.items()}
    return res


def _draws(E_reps, output, P_w, R_w, pers_den, names):
    C = E_reps.mean(axis=2) if output == 'composite' else E_reps[:, :, names.index(output)]
    num = C @ P_w.T / C.shape[1]                                 # (R, B)
    num = (R_w * num.T).sum(axis=1) / C.shape[0] if R_w is not None else num[0]
    return num / pers_den


def bootstrap_table(res, output='composite', n_boot=2000, seed=0):
    """Expected relative error of each strategy at each size, its 95% CI, and the
    paired difference from the reference with a verdict."""
    rng = np.random.default_rng(seed)
    names = res.output_names or [f'y{j}' for j in range(res.pers_errors.shape[1])]
    P = res.pers_errors.shape[0]
    P_w = np.stack([np.bincount(rng.integers(0, P, P), minlength=P) for _ in range(n_boot)]).astype(float)
    base = res.pers_errors.mean(axis=1) if output == 'composite' else res.pers_errors[:, names.index(output)]
    pers_den = P_w @ base / P
    ref = _draws(res.ref_errors[None], output, P_w, None, pers_den, names)
    rows = []
    for n in res.sizes:
        R = next(iter(res.errors[n].values())).shape[0]
        R_w = np.stack([np.bincount(rng.integers(0, R, R), minlength=R) for _ in range(n_boot)]).astype(float)
        for name, E in res.errors[n].items():
            t = _draws(E, output, P_w, R_w, pers_den, names)
            d = t - ref
            lo, hi = np.percentile(t, [2.5, 97.5])
            dlo, dhi = np.percentile(d, [2.5, 97.5])
            verdict = 'reference better' if dlo > 0 else ('strategy better' if dhi < 0 else 'indistinguishable')
            rows.append(dict(n=n, strategy=name, rel_error=float(t.mean()), lo=lo, hi=hi,
                             diff=float(d.mean()), diff_lo=dlo, diff_hi=dhi, verdict=verdict))
    return pd.DataFrame(rows)


def threshold_with_ci(res, strategy, output='composite', n_boot=1000, seed=0):
    """Durable threshold for beating persistence, with a bootstrap 95% CI snapped
    to grid sizes and the share of resamples that never cross."""
    rng = np.random.default_rng(seed)
    names = res.output_names or [f'y{j}' for j in range(res.pers_errors.shape[1])]
    P = res.pers_errors.shape[0]
    P_w = np.stack([np.bincount(rng.integers(0, P, P), minlength=P) for _ in range(n_boot)]).astype(float)
    base = res.pers_errors.mean(axis=1) if output == 'composite' else res.pers_errors[:, names.index(output)]
    pers_den = P_w @ base / P
    curves = []
    for n in res.sizes:
        E = res.errors[n][strategy]
        R = E.shape[0]
        R_w = np.stack([np.bincount(rng.integers(0, R, R), minlength=R) for _ in range(n_boot)]).astype(float)
        curves.append(_draws(E, output, P_w, R_w, pers_den, names))
    curves = np.column_stack(curves)
    point = durable_threshold(res.sizes, curves.mean(axis=0))
    sb = np.array([durable_threshold(res.sizes, c) for c in curves])
    fin = sb[np.isfinite(sb)]
    lo, hi = (np.percentile(fin, [2.5, 97.5], method='nearest') if len(fin) > 10 else (np.nan, np.nan))
    return dict(threshold=point, lo=float(lo), hi=float(hi), never=float(np.isnan(sb).mean()))


def recommend(table):
    """Plain-language summary: for each strategy, the largest size at which it is
    still worse than the reference and the first size from which it is better."""
    out = {}
    for name, g in table.groupby('strategy'):
        g = g.sort_values('n')
        worse = g[g.verdict == 'reference better'].n
        better = g[g.verdict == 'strategy better'].n
        out[name] = dict(worse_up_to=int(worse.max()) if len(worse) else None,
                         better_from=int(better.min()) if len(better) else None)
    return out
