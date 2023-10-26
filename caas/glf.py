import warnings

import numpy as np

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)


def generalised_logistic_function(t: np.ndarray, a: np.float64, k: np.float64,
                                  c: np.float64, b: np.float64, v: np.float64,
                                  q: np.float64):
    """
    Generalised logistic function.

    t: load (in requests per second)
    ...
    """
    return a + ((k - a) / (c + q * np.exp(-b * t)) ** (1 / v))


def speed_up(n: float, s: float):
    """
    Amdahl's law: The speedup depends on the amount of code that cannot be 
    parallelized.

    n: number of processors
    s: percentage of code that is inherently sequential
    """
    return 1 / (s + ((1 - s) / n))


def g_generalised_logistic_function(X, a: np.float64, k: np.float64,
                                    c: np.float64, b: np.float64, v: np.float64,
                                    q: np.float64):
    """
    G-Generalised logistic function.

    t: load (in requests per second)
    r: number of replicas
    s: percentage of code that cannot be made parallel
    ...
    """
    t, r, s = X
    return a + ((k - a) / (c + q * np.exp(-b * (t / speed_up(r, s)))) ** (1 / v))
