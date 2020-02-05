""" TGARCH estimation
Sorry for the bad notation.
"""
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import scipy.optimize as opt
from arch import arch_model
from arch.univariate import EGARCH
from statsmodels.tsa.stattools import acf
import pickle



#%%
def acf(x, h):
    n = len(x)
    xbar = np.average(x)
    xvar = np.sum([(xi-xbar)**2 for xi in x])
    x0 = x[0:n-h]
    x1 = x[h:]
    xcov = np.sum([(x0[i]-xbar)*(x1[i]-xbar) for i in range(len(x0))])
    return xcov/xvar

#%%
def Qstat(x, m):
    # print('Q test')
    n = len(x)
    Q = np.sum([(acf(x, i) ** 2) / (n - i) for i in range(1, m + 1)]) * n * (n + 2)
    # print(Q)
    pval = 1 - stats.chi2.cdf(Q, m)
    return Q, pval


#%%
def f_s2t(omega, alpha, gamma, beta, x0, s20):
    var = omega + ((alpha+gamma*np.float(x0 < 0))*(x0**2)) + (beta*s20)
    # print((omega, alpha, gamma, beta, x0, s20))
    if var < 0:
        print('NEGATIVE VARIANCE')
    return var


def build_s2(xx, omega, alpha, gamma, beta):  #**kwargs):
    # xx: list
    # Initial value:
    xx2 = [x**2 for x in xx]
    l_s2 = [np.average(xx2)]
    for i in range(len(xx)-1):
        # if l_s2[i] == np.inf:
        #     print(omega, alpha, gamma, beta, xx[i], l_s2[i])
        l_s2 += [f_s2t(omega, alpha, gamma, beta, xx[i], l_s2[i])]  # **kwargs)]
    return l_s2


def ll(s2t, xt):
    return -np.log(s2t) - ((xt**2)/s2t)


def avg_ll(xx, omega, alpha, gamma, beta):
    l_s2 = build_s2(xx, omega, alpha, gamma, beta)
    LL = np.average([ll(l_s2[i], xx[i]) for i in range(len(xx))])
    LL = (-0.5 * np.log(2 * np.pi)) + 0.5 * LL
    # print(LL)
    return -LL


cons = [{'type': 'ineq', 'fun': lambda theta:  theta[1] + (theta[2]/2) + theta[3]+1},
        {'type': 'ineq', 'fun': lambda theta:  1 - (theta[1] + (theta[2]/2) + theta[3])}]

# cons = ({'type': 'ineq', 'fun': lambda theta:  0.99-abs(theta[0] - theta[1])}, # |alpha-beta|<= 0.99
#         {'type': 'ineq', 'fun': lambda theta:  10*4-abs(theta[2])}, # |omega| <= 10^4
#         {'type': 'ineq', 'fun': lambda theta:  theta[3]-10**(-4)}, # delta >= 10^(-4)
#         {'type': 'ineq', 'fun': lambda theta:  10-theta[3]}, # delta <= 10
#         {'type': 'ineq', 'fun': lambda theta:  theta[4]-5}, # lambda >= 5
#         {'type': 'ineq', 'fun': lambda theta:  100-theta[4]}, # lambda <= 100
#         {'type': 'ineq', 'fun': lambda theta:  theta[5]-10**(-4)}, # sigma >= 10^(-4)
#         {'type': 'ineq', 'fun': lambda theta:  10**4-theta[5]}) # sigma <= 10^4)

# x: array([ 2.14681928e-06, -8.00922261e-03,  1.75349404e-01,  8.96252110e-01])


def MLE(xx, theta0=(0.6, 0.25, 0.9, 0.2)):
    res = opt.minimize(lambda theta: avg_ll(xx, theta[0], theta[1], theta[2], theta[3]), theta0,
                        # method='Nelder-Mead')
                        # method='BFGS')
                        method='SLSQP',
                        constraints=cons,
                        bounds=[(0, 10.0),(0, 10.0),(0, 10.0), (0, 10.0)])
    # res = opt.fmin_slsqp(lambda theta: avg_ll(xx, theta[0], theta[1], theta[2], theta[3]), theta0,
    #                      f_ieqcons=lambda theta:  (1 - (theta[1] + (theta[2]/2) + theta[3])),
    #                      bounds=[(0.001, 10.0),(0.0, 1.0),(0.0, 1.0), (0.0, 1.0)])
                       # args = args)

    return res

""" Test """
# xx = [1, 2, 4, 6, 1, 3, 5, 7]
# omega = 1
# alpha = 0.2
# beta = 0.6
# gamma = 0.1
# l_s2 = build_s2(xx, omega=omega, alpha=alpha, gamma=gamma, beta=beta)

# acf(xx, 2)
# print(Qstat(xx, 3))
# quit()

""" Actual estimation """
# print(csv.reader(open('oxfordmanrealizedvolatilityindices.csv', 'rb')))
dta = pd.read_csv('oxfordmanrealizedvolatilityindices.csv')
index = '.FTSE'

dta = dta[['Unnamed: 0', 'Symbol', 'open_to_close', 'rv5', 'bv', 'rk_parzen']]
dta = dta[dta.Symbol.eq(index)]
dta.columns = ['date', 'S', 'r', 'rv', 'bv', 'rk']

XX = list(dta.r)
XX = [x*100 for x in XX]
res = MLE(XX)

print(res)


with open("Q2_LL.txt", "wb") as fp:   #Pickling
    pickle.dump(res.fun*(len(XX)-1), fp)

# Get fitted value of sigma2:
#%%

fitted_s2 = build_s2(XX, res.x[0], res.x[1], res.x[2], res.x[3])
fitted_Z = [XX[i]/np.sqrt(fitted_s2[i]) for i in range(len(XX))]
fitted_absZ = [abs(Z) for Z in fitted_Z]
fitted_Z2 = [Z**2 for Z in fitted_Z]
pvals_Z = [Qstat(fitted_Z, m+1) for m in range(10)]
pvals_absZ = [Qstat(fitted_absZ, m+1) for m in range(10)]
pvals_Z2 = [Qstat(fitted_Z2, m+1) for m in range(10)]
#%%

print(pvals_Z)
print(pvals_absZ)
print(pvals_Z2)

with open("h_hat_Q2.txt", "wb") as fp:   #Pickling
    pickle.dump(fitted_s2, fp)

plt.plot([i+1 for i in range(len(fitted_s2))], fitted_s2)
plt.savefig('fitted_h.png')
plt.clf()


nlags = 20
autocorr = [pd.Series(fitted_Z2).autocorr(lag) for lag in range(nlags)]


model = arch_model(100*dta.r, mean='Zero', p=1, q=1, o=1)
result = model.fit(disp='off')
print(result.summary())

# result.aic
l_result = [arch_model(100*dta.r,mean='Zero', p=i, q=1, o=1).fit(disp='off').bic for i in range(1, 5)]


model = arch_model(100*dta.r, p=1, q=1, o=1, dist='t') #mean='Zero' dist='StudentsT',
result = model.fit(disp='off')
print(result.summary())
residuals = list(result.std_resid)

pvals_resid = [Qstat(residuals, m+1)[1] for m in range(10)]
print(pvals_resid)

print('JARQUE-BERA:')
print(stats.jarque_bera(residuals))
print(stats.kstest(residuals, 't', args=(10, )))

# fig, ax = plt.subplots()
# plt.hist(residuals, 100)
stats.probplot(residuals, dist='t', plot=plt, sparams=(10,))
plt.savefig('Q2_QQ_t.png')
# plt.show()

plt.clf()
stats.probplot(residuals, dist='norm', plot=plt)
# ax.get_lines()[0].set_marker('p')
# ax.get_lines()[0].set_markerfacecolor('r')
plt.savefig('Q2_QQ_norm.png')
# plt.show()

# egarch = EGARCH(, p=1, o=1, q=1)

l_result_norm_t = [arch_model(100*dta.r, dist=i, p=1, q=1, o=1).fit(disp='off').bic for i in ['Normal', 't']]


model = arch_model(100*dta.r, x=dta.rv, mean='Constant', p=1, q=1, o=0, dist='t') #mean='Zero' dist='StudentsT',
result = model.fit(disp='off')
print(result.summary())
