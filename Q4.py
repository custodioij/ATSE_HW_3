import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scipy.optimize as opt
from arch import arch_model
from arch.univariate import EGARCH
from statsmodels.tsa.stattools import acf
import pickle


dta = pd.read_csv('oxfordmanrealizedvolatilityindices.csv')
index = '.FTSE'

dta = dta[['Unnamed: 0', 'Symbol', 'open_to_close', 'rv5', 'bv', 'rk_parzen']]
dta = dta[dta.Symbol.eq(index)]
dta.columns = ['date', 'S', 'r', 'rv', 'bv', 'rk']

# XX = list(dta.r)
rv = list(dta.rv)
lrv = list(np.log(rv))


def build_lags(x, k):
    """
    k: list of lags
    """
    xl = [x[max(k):]]
    for i in k:
        xl += [x[(max(k)-i):-i]]
    # print(xl)
    return np.array(xl).T

dta_rv = build_lags(rv, [1, 5, 22])
dta_lrv = build_lags(lrv, [1, 5, 22])


