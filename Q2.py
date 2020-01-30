""" TGARCH estimation """
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# print(csv.reader(open('oxfordmanrealizedvolatilityindices.csv', 'rb')))
dta = pd.read_csv('oxfordmanrealizedvolatilityindices.csv')
index = '.FTSE'

dta = dta[['Unnamed: 0', 'Symbol', 'open_to_close', 'rv5', 'bv', 'rk_parzen']]
dta = dta[dta.Symbol.eq(index)]
dta.columns = ['date', 'S', 'r', 'rv', 'bv', 'rk']

