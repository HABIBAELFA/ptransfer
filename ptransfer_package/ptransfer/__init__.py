"""ptransfer: persistence-anchored evaluation of imported longitudinal prediction models.

Tells a centre, from its own data, whether and when adapting an imported
model improves on leaving it unchanged, measured against persistence
(the prediction that nothing changes)."""
from .core import (wrap, default_scale, patient_errors, relative_error, durable_threshold,
                   source_r2, rank_agreement)
from .strategies import Frozen, Recalibration, LocalModel, ResidualBoosting, cv_shrink_weight
from .compare import compare_strategies, bootstrap_table, threshold_with_ci, recommend

__version__ = '0.1.0'
__all__ = ['wrap', 'default_scale', 'patient_errors', 'relative_error', 'durable_threshold',
           'source_r2', 'rank_agreement', 'Frozen', 'Recalibration', 'LocalModel',
           'ResidualBoosting', 'cv_shrink_weight', 'compare_strategies', 'bootstrap_table',
           'threshold_with_ci', 'recommend']
