""" TGARCH estimation
Sorry for the bad notation.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt


def f_s2t(omega, alpha, gamma, beta, x0, s20):
    var = omega + ((alpha + gamma*np.float(x0 <= 0))*(x0**2)) + (beta*s20)
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
    print(LL)
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


def MLE(xx, theta0=(0.1, 0.25, 0.9, 0.2)):
    res = opt.minimize(lambda theta: avg_ll(xx, theta[0], theta[1], theta[2], theta[3]), theta0,
                        # method='Nelder-Mead')
                        # method='BFGS')
                        method='SLSQP',
                        constraints=cons,
                        bounds=[(-1e10, 10.0),(-1.0, 10.0),(0.0, 10.0), (0.0, 10.0)])
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
# # l_s2 = build_s2(xx, omega=omega, alpha=alpha, gamma=gamma, beta=beta)
# LL = avg_ll(xx, omega, alpha, gamma, beta)
# print(LL)
# mle = MLE(xx)
# print(mle)
#
# quit()

""" Actual estimation """
# print(csv.reader(open('oxfordmanrealizedvolatilityindices.csv', 'rb')))
dta = pd.read_csv('oxfordmanrealizedvolatilityindices.csv')
index = '.FTSE'

dta = dta[['Unnamed: 0', 'Symbol', 'open_to_close', 'rv5', 'bv', 'rk_parzen']]
dta = dta[dta.Symbol.eq(index)]
dta.columns = ['date', 'S', 'r', 'rv', 'bv', 'rk']

XX = list(dta.r)
# XX = [x+1 for x in XX]
# print(np.var(XX))
# XX
res = MLE(XX)

print(res)

