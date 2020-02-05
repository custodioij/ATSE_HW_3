"""
Code for question 1 of the assignment.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt
import math as math
from scipy. stats import kstest
import scipy.stats as stats
from scipy.special import gamma
import pickle


# print(csv.reader(open('oxfordmanrealizedvolatilityindices.csv', 'rb')))
dta = pd.read_csv('oxfordmanrealizedvolatilityindices.csv')
index = '.FTSE'

dta = dta[['Unnamed: 0', 'Symbol', 'open_to_close', 'rv5', 'bv', 'rk_parzen']]
dta = dta[dta.Symbol.eq(index)]
dta.columns = ['date', 'S', 'r', 'rv', 'bv', 'rk']

#print(dta.shape)
#print(np.mean(np.abs(np.divide(dta.r, np.sqrt(dta.bv))) > 1.96))
# Jumps 11% of the time, 11% of 5062 days



#print(np.mean(dta.rv))
#print(np.mean(dta.rk))
# quit()

# fig = plt.figure()
dta[['rv', 'bv', 'rk']].plot(alpha=0.5, subplots=True, sharey=True)
# plt.show()
# plt.savefig('Q1.png')  # TODO: change x-labels to date (don't know how).
# quit()


def volatility(theta, x, h, rv):
    omega, beta, alpha, gamma1, delta, nu = theta
    v= omega + (alpha + gamma1 * (x < 0)) * x ** 2 + beta * h + delta * rv
    return v

def filter(mX, theta):
    iN = np.shape(mX)[0]
    vH = np.zeros(iN + 1)
    vH[0] = np.average(mX[:,0]**2)
    for i in range(iN):
        vH[i+1] = volatility(theta, mX[i, 0], vH[i], mX[i, -1])
    return vH
#%%
def garch2(theta, mX, dist):
    iN = np.shape(mX)[0]
    omega, beta, alpha, gamma1, delta, nu = theta
    LL = np.zeros(iN)
    vH= filter(mX, theta)
    if dist == 'norm':
        LL = np.log(vH[:-1]) + (mX[:, 0]**2)/(vH[:-1])
        loglik = 0.5*iN*np.log(2*np.pi)+0.5*np.sum(LL)
    elif dist == 't':
        xx = (mX[:, 0]**2)/vH[:-1]
        LL = 0.5*(nu+1)*np.log((1+xx/nu)) + np.log(np.sqrt(vH[:-1]))
        loglik = -iN * np.log(math.gamma((nu + 1) / 2)) + np.sum(LL) + 0.5 * iN * np.log(nu * np.pi) + iN * np.log(
            math.gamma(nu / 2))
    return loglik

#%%
#theta_ini = 0.3*np.ones(5)

dta1 = np.array(dta.drop(['S', 'date'], axis=1))
theta_ini = [0.01, 0.9, 0.5, 0.3, 1, 10]
cons = ({'type': 'ineq', 'fun': lambda theta:  0.99999-(theta[1] + theta[2]+theta[3]/2)},
        {'type': 'ineq', 'fun': lambda theta:  (theta[1] + theta[2]+theta[3]/2)})
dist = 't'
results = opt.minimize(garch2, theta_ini, args=(dta1*100, dist), method='SLSQP', constraints=cons,
                       bounds=[(0, 10), (0, 10), (0.0, 10), (0, 10), (-100.0, 100.0), (2, 200)])
print(results)

h_hat_Q3 = filter(dta1*100, results.x)
with open("h_hat_Q3.txt", "wb") as fp:   #Pickling
    pickle.dump(h_hat_Q3, fp)

print('Normal:')
results_n = opt.minimize(garch2, theta_ini, args=(dta1*100, 'norm'), method='SLSQP', constraints=cons,
                       bounds=[(0, 10), (0, 10), (0.0, 10), (0, 10), (-100.0, 100.0), (2, 200)])
print(results_n)

with open("Q3_LL.txt", "wb") as fp:   #Pickling
    pickle.dump(results_n.fun, fp)

sigma2 = filter(dta1*100, results.x)
resid = dta1[100:,0]/np.sqrt(sigma2[100:-1])
plt.plot(resid)
plt.show()
plt.hist(resid, bins= 40)
plt.show()
kstest(resid,'t', args= (13,))
stats.probplot(resid, dist= 't', plot=plt, sparams= (13,))
plt.show()
