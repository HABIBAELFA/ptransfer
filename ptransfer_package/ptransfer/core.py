"""Persistence-anchored evaluation of longitudinal predictions.

All functions work on arrays of transitions:
    D        (n, k)  observed change of k outputs between two visits
    P        (n, k)  predicted change, in the same units as D
    subject  (n,)    patient identifier of each transition
    circular (k,)    boolean mask of outputs measured in degrees (wrapped to +/-180)

Predicting a change of zero is persistence ("nothing changes"), so the
comparator is always available and needs no data.
"""
import numpy as np
from scipy import stats


def wrap(x):
    """Wrap angles in degrees to [-180, 180)."""
    return (np.asarray(x, float) + 180.0) % 360.0 - 180.0


def _abs_residual(D, P, circular):
    R = np.asarray(D, float) - np.asarray(P, float)
    if circular is not None:
        c = np.asarray(circular, bool)
        if c.any():
            R[:, c] = wrap(R[:, c])
    return np.abs(R)


def default_scale(D, circular=None):
    """Per-output scale used to put outputs on a common footing in the composite.
    It cancels in every per-output relative error."""
    D = np.array(D, float)
    if circular is not None and np.any(circular):
        D[:, np.asarray(circular, bool)] = wrap(D[:, np.asarray(circular, bool)])
    s = D.std(axis=0, ddof=1)
    s[~np.isfinite(s) | (s <= 0)] = 1.0
    return s


def patient_errors(D, P, subject, circular=None, scale=None):
    """Mean absolute error per patient and output, shape (n_patients, k).

    Averaging within patient first stops frequently recorded patients from
    dominating the result. Returns (errors, patient_ids)."""
    A = _abs_residual(D, P, circular)
    if scale is not None:
        A = A / np.asarray(scale, float)
    ids, inv = np.unique(np.asarray(subject), return_inverse=True)
    sums = np.zeros((len(ids), A.shape[1]))
    np.add.at(sums, inv, A)
    counts = np.bincount(inv, minlength=len(ids)).astype(float)
    return sums / counts[:, None], ids


def relative_error(D, P, subject, circular=None, scale=None):
    """Relative error against persistence.

    Returns a dict with 'composite' (float) and 'per_output' (array of k).
    Values below 1 improve on predicting no change."""
    scale = default_scale(D, circular) if scale is None else scale
    E, _ = patient_errors(D, P, subject, circular, scale)
    E0, _ = patient_errors(D, np.zeros_like(np.asarray(D, float)), subject, circular, scale)
    return {'composite': float(E.mean(axis=1).mean() / E0.mean(axis=1).mean()),
            'per_output': E.mean(axis=0) / E0.mean(axis=0)}


def durable_threshold(sizes, curve, target=1.0):
    """Smallest size from which the curve stays below `target` at every larger
    size. Returns np.nan when the curve never settles below the target."""
    sizes = np.asarray(sizes, float)
    below = np.asarray(curve, float) < target
    for i in range(len(sizes)):
        if below[i:].all():
            return float(sizes[i])
    return float('nan')


def source_r2(X, D, circular=None):
    """In-sample R^2 of a least-squares model of change (with intercept) for each
    output, computed on the SOURCE cohort only. Used to rank outputs by the
    number of local patients they are likely to need."""
    X = np.column_stack([np.ones(len(X)), np.asarray(X, float)])
    D = np.array(D, float)
    if circular is not None and np.any(circular):
        D[:, np.asarray(circular, bool)] = wrap(D[:, np.asarray(circular, bool)])
    out = []
    for j in range(D.shape[1]):
        y = D[:, j]
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        r = y - X @ coef
        out.append(1.0 - r.var(ddof=1) / y.var(ddof=1))
    return np.array(out)


def rank_agreement(r2, thresholds):
    """Spearman correlation between source R^2 and observed thresholds.
    A strongly negative value means R^2 ranks the outputs correctly."""
    ok = np.isfinite(np.asarray(thresholds, float))
    rho, p = stats.spearmanr(np.asarray(r2)[ok], np.asarray(thresholds)[ok])
    return float(rho), float(p)
