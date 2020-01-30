"""
Code for question 1 of the assignment.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# print(csv.reader(open('oxfordmanrealizedvolatilityindices.csv', 'rb')))
dta = pd.read_csv('oxfordmanrealizedvolatilityindices.csv')
index = '.FTSE'

dta = dta[['Unnamed: 0', 'Symbol', 'open_to_close', 'rv5', 'bv', 'rk_parzen']]
dta = dta[dta.Symbol.eq(index)]
dta.columns = ['date', 'S', 'r', 'rv', 'bv', 'rk']

print(dta.shape)
print(np.mean(np.abs(np.divide(dta.r, np.sqrt(dta.bv))) > 1.96))
# Jumps 11% of the time, 11% of 5062 days

quit()

print(np.mean(dta.rv))
print(np.mean(dta.rk))
# quit()

# fig = plt.figure()
dta[['rv', 'bv', 'rk']].plot(alpha=0.5, subplots=True, sharey=True)
plt.show()

print('hi masha')
