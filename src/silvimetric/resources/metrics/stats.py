import numpy as np
from scipy import stats
from ..metric import Metric
from .p_moments import mean
from .percentiles import pct_base

import warnings
# suppress warnings from dividing by 0, these are handled in the metric creation
warnings.filterwarnings(
    action='ignore',
    category=RuntimeWarning,
    module='lmoments3'
)

def m_mode(data):
    u, c = np.unique(data, return_counts=True)
    i = np.where(c == c.max())
    v = u[i[0][0]]
    return v

def m_median(data, *args):
    a = np.median(data)
    return a

def m_min(data, *args):
    return np.min(data)

def m_max(data, *args):
    return np.max(data)

def m_stddev(data, *args):
    return np.std(data)

def m_cv(data, *args):
    stddev, mean = args
    if mean == 0:
        return np.nan
    return stddev / mean

# TODO check performance of other methods
def m_abovemean(data, *args):
    mean = args[0]
    l = len(data)
    if l == 0:
        return np.nan
    return (data > mean).sum() / l

# TODO check performance of other methods
def m_abovemode(data, *args):
    mode = args[0]
    l = len(data)
    if l == 0:
        return np.nan
    return (data > mode).sum() / l

def m_iq(data):
    return stats.iqr(data)

def m_crr(data, *args):
    mean, minimum, maximum = args
    den = (maximum - minimum)
    if den == 0:
        return np.nan
    return (mean - minimum) / den

def m_sqmean(data):
    return np.sqrt(np.mean(np.square(data)))

def m_cumean(data):
    return np.cbrt(np.mean(np.power(np.absolute(data), 3)))

def m_profilearea(data, *args):
    # sanity check...must have valid heights/elevations
    dmax, dmin , p = args
    if dmax <= 0:
        return -9999.0

    # p = np.percentile(data, range(1, 100))
    p0 = max(dmin, 0.0)

    # second sanity check...99th percentile must be > 0
    p99 = p[99]
    if p99 > 0.0:
        # compute area under normalized percentile height curve using composite trapeziod rule
        pcts = np.array(p[:98])
        areas = pcts * 2 / p99
        pa = p0/p99 + areas.sum() + 1

        return pa * 0.5
    else:
        return -9999.0

# # TODO example for cover using all returns and a height threshold
# # the threshold must be a parameter and not hardcoded
# def m_cover(data):
#     threshold = 2
#     return (data > threshold).sum() / len(data)

mode = Metric('mode', np.float32, m_mode)
median = Metric('median', np.float32, m_median)
# TODO better names?
sm_min = Metric('min', np.float32, m_min)
sm_max = Metric('max', np.float32, m_max)
stddev = Metric('stddev', np.float32, m_stddev)
cv = Metric('cv', np.float32, m_cv, [ mean, stddev ])
abovemean = Metric('abovemean', np.float32, m_abovemean, [ mean ])
abovemode = Metric('abovemode', np.float32, m_abovemode, [ mode ])
iq = Metric('iq', np.float32, m_iq)
crr = Metric('crr', np.float32, m_crr, [ mean, sm_min, sm_max ])
sqmean = Metric('sqmean', np.float32, m_sqmean)
cumean = Metric('cumean', np.float32, m_cumean)
profilearea = Metric('profilearea', np.float32, m_profilearea, [ sm_max, sm_min, pct_base ])

statistics: dict[str, Metric] = dict(
    mode=mode,
    median=median,
    min=sm_min,
    max=sm_max,
    stddev=stddev,
    cv=cv,
    abovemean=abovemean,
    abovemode=abovemode,
    iq=iq,
    crr=crr,
    sqmean=sqmean,
    cumean=cumean,
    profilearea=profilearea,
)
# statistics['cover'] = Metric('cover', np.float32, m_cover)
