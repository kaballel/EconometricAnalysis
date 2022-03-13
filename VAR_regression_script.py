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

# Initial Data Visualization
# df.plot(subplots=True, figsize=(6, 6)); plt.legend(loc='best')
# plt.show()

print(df)
