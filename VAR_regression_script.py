# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.tools.eval_measures import rmse, aic

# Variable Names for Reference:
# pop, prodElectric, lenRail, agrProd, numPatents, numMiners, numCorps, percAgr, percManu, gdpPerCap


# Import Data for Analysis
pd.read_csv('./Primary_Dataset.csv')
