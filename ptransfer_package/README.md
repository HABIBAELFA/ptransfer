# ptransfer

Archived on Zenodo (all versions): [10.5281/zenodo.22998860](https://doi.org/10.5281/zenodo.22998860)

**Should a centre adapt an imported prediction model, and from how many local patients?**

`ptransfer` evaluates longitudinal prediction models against *persistence* (the
prediction that nothing changes), a comparator that needs no data, and compares
transfer strategies for an imported model across local cohort sizes, with
confidence intervals that account for both which local patients a centre has and
the test sample.

It accompanies the paper *When does adapting an imported prediction model help?
Persistence-anchored transfer of longitudinal ECG prediction across departments
and devices* (manuscript under review).

## What it does

| Function | Purpose |
|---|---|
| `relative_error(D, P, subject, circular)` | Error relative to persistence, averaged within patient; per output and composite |
| `Frozen`, `Recalibration`, `LocalModel`, `ResidualBoosting` | Transfer strategies with `fit(X, D, subject)` / `predict(X)`; shrinkage toward the safe default by cross-validation grouped by patient |
| `compare_strategies(...)` | Fits every strategy on the same random draws of *n* local patients |
| `bootstrap_table(res)` | Expected relative error, 95% CI, and paired difference from the imported model (two-level bootstrap) |
| `threshold_with_ci(res, strategy)` | Local patients needed to beat persistence durably, with CI |
| `source_r2(X, D)` | Source-only R² that ranks outputs by the local data they will need |
| `recommend(table)` | Plain-language summary: worse up to *n*, better from *n* |

Works with any scikit-learn-style regressor. Outputs measured in degrees
(e.g. ECG axes) are handled circularly.

## Quick start

```python
import ptransfer as pt
strategies = {
    'frozen': lambda: pt.Frozen(source_predict),
    'residual boosting, shrunk': lambda: pt.ResidualBoosting(source_predict, gbm, shrink=True),
}
res = pt.compare_strategies(strategies, X_pool, D_pool, subj_pool, X_test, D_test, subj_test,
                            sizes=[20, 100, 1000], n_draws=10)
table = pt.bootstrap_table(res)
print(pt.recommend(table))
```

`examples/demo_synthetic.py` runs a complete example on a synthetic cohort in about one minute. `examples/parametric_simulation.py` reproduces the parametric simulation of the paper (20 scenarios x 3 seeds, about 20 min on two cores).

## Install and test

```
pip install -e .
pytest tests
```

Requires Python ≥ 3.9, NumPy, SciPy, pandas, scikit-learn.

## Licence

MIT.
