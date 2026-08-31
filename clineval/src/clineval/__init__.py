"""
clineval — calibration, discrimination, and decision curve analysis
for binary clinical prediction models.

Quickstart:

    import clineval as ce

    ce.calibration.calibration_summary(y_true, y_prob)
    ce.discrimination.discrimination_summary(y_true, y_prob)
    ce.decision_curve.net_benefit(y_true, y_prob)
"""

from . import calibration
from . import discrimination
from . import decision_curve

try:
    from . import plotting  # optional: requires matplotlib
except ImportError:
    plotting = None

__version__ = "0.1.0"

__all__ = ["calibration", "discrimination", "decision_curve", "plotting", "__version__"]
