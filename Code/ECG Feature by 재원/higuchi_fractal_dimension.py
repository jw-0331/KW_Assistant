#!/usr/bin/env python
# coding: utf-8

# In[1]:



# Compatibility imports
from __future__ import absolute_import, division, print_function

# 3rd party imports
import numpy as np


def hfd(a, k_max):

    """Compute Higuchi Fractal Dimension of a time series."""

    # Initialize variables
    L = []
    x = []
    N = a.size

    for k in range(1, k_max):
        Lk = 0
        for m in range(0, k):
            idxs = np.arange(1, int(np.floor((N - m) / k)), dtype=np.int32)
            Lmk = np.sum(np.abs(a[m + idxs * k] - a[m + k * (idxs - 1)]))
            Lmk = (Lmk * (N - 1) / (((N - m) / k) * k)) / k
            Lk += Lmk
        L.append(np.log(Lk / (m + 1)))
        x.append([np.log(1.0 / k), 1])
    (p, r1, r2, s) = np.linalg.lstsq(x, L)

    return p[0]

