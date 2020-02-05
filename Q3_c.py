import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scipy.optimize as opt
from arch import arch_model
from arch.univariate import EGARCH
from statsmodels.tsa.stattools import acf
import pickle

with open("h_hat_Q2.txt", "rb") as fp:   # Unpickling
    h_TGARCH = pickle.load(fp)

with open("h_hat_Q3.txt", "rb") as fp:   # Unpickling
    h_realized = pickle.load(fp)

h_realized = list(h_realized)

dta = pd.read_csv('oxfordmanrealizedvolatilityindices.csv')
index = '.FTSE'

dta = dta[['Unnamed: 0', 'Symbol', 'open_to_close', 'rv5', 'bv', 'rk_parzen']]
dta = dta[dta.Symbol.eq(index)]
dta.columns = ['date', 'S', 'r', 'rv', 'bv', 'rk']

# XX = list(dta.r)
rv = list(dta.rv*(100**2))

pd.DataFrame(list(zip(h_TGARCH, h_realized, rv)), columns=['h_TGARCH', 'h_realized', 'rv']).plot(alpha=0.5, subplots=True, sharey=True)
# plt.show()
plt.savefig('Q3_c.png')

mse_2 = np.mean([(h_TGARCH[i] - rv[i]) ** 2 for i in range(len(rv))])
mse_3 = np.mean([(h_realized[i] - rv[i]) ** 2 for i in range(len(rv))])


""" LL ratio """

with open("Q2_LL.txt", "rb") as fp:   # Unpickling
    LL_2 = pickle.load(fp)  # Constrained

with open("Q3_LL.txt", "rb") as fp:   # Unpickling
    LL_3 = pickle.load(fp)  # Unconstrained

LR = (LL_2 - LL_3)*2