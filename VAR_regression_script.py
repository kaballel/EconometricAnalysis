# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

# Variable Names for Reference:
# pop, prodElectric, lenRail, prodAgr, numPatents, numMiners, numCorps, percAgr, percManu, gdpPerCap


# Import Data for Analysis
df = pd.read_csv('./Primary_Dataset_1_CSV.csv', parse_dates=['Year'], index_col=['Year'])
df.drop(['Unnamed: 11', 'Unnamed: 12'], axis=1, inplace=True)
df.astype(float)

# Initial Data Visualization
print(df.head())
print('\n')

df['1903-01-01':'2003-01-01'].plot()
# df.plot(x='Year', y='pop')
# plt.show()
df.plot(y='pop')
plt.show()
