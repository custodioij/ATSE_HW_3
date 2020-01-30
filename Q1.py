"""
Code for question 1 of the assignment.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize as opt

# print(csv.reader(open('oxfordmanrealizedvolatilityindices.csv', 'rb')))
dta = pd.read_csv('oxfordmanrealizedvolatilityindices.csv')
index = '.FTSE'

dta = dta[['Unnamed: 0', 'Symbol', 'open_to_close', 'rv5', 'bv', 'rk_parzen']]
dta = dta[dta.Symbol.eq(index)]
dta.columns = ['date', 'S', 'r', 'rv', 'bv', 'rk']

print(dta.shape)
print(np.mean(np.abs(np.divide(dta.r, np.sqrt(dta.bv))) > 1.96))
# Jumps 11% of the time, 11% of 5062 days



print(np.mean(dta.rv))
print(np.mean(dta.rk))
# quit()

# fig = plt.figure()
dta[['rv', 'bv', 'rk']].plot(alpha=0.5, subplots=True, sharey=True)
plt.show()


#%%
def garch2(theta, mX):
    iN = mX.shape[0]
    vH = np.zeros(iN+1)
    omega, beta, alpha, gamma, delta = theta
    vH[0] = omega/(1-alpha-beta-0.5*gamma)
    LL = np.zeros(iN)
    for i in range(iN):
        vH[i+1] = omega + (alpha + gamma * (mX.r.iloc[i] < 0))*mX.r.iloc[i]**2 + beta * vH[i] + delta * mX.rv.iloc[i]
        LL[i] = np.log(vH[i]) + (mX.r.iloc[i]**2)/(vH[i])
    print(np.mean(vH))
    return np.mean(LL)

#%%
#theta_ini = 0.3*np.ones(5)


theta_ini = [0.001, 0.5, 0.2, 0.1, 0.3]
cons = ({'type': 'ineq', 'fun': lambda theta:  0.99999-abs(theta[1] + theta[2]+theta[3]/2)},
        {'type': 'ineq', 'fun': lambda theta:  theta[0]+1},
        {'type': 'ineq', 'fun': lambda theta:  theta[1]+1},
        {'type': 'ineq', 'fun': lambda theta:  theta[2]+1},
        {'type': 'ineq', 'fun': lambda theta:  theta[3]+1},
        {'type': 'ineq', 'fun': lambda theta:  theta[4]+1},
        {'type': 'ineq', 'fun': lambda theta:  10 - theta[0]},
        {'type': 'ineq', 'fun': lambda theta:  10 - theta[1]},
        {'type': 'ineq', 'fun': lambda theta:  10 - theta[2]},
        {'type': 'ineq', 'fun': lambda theta:  10 - theta[3]},
        {'type': 'ineq', 'fun': lambda theta:  10 - theta[4]})
results = opt.minimize(garch2, theta_ini, args=dta, method='SLSQP', constraints=cons)
