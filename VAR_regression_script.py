# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import statsmodels.api as sm
from pprint import pprint

from statsmodels.tsa.api import VAR
from statsmodels.tsa.stattools import adfuller
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tools.eval_measures import rmse, aic
from statsmodels.tsa.vector_ar.vecm import coint_johansen
from statsmodels.tsa.vector_ar import irf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.ar_model import ar_select_order, AutoReg
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.stattools import grangercausalitytests

# Ignore Warnings on Code (ATS throwing errors), Set Pandas Options
warnings.filterwarnings("ignore")
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Import Data for Analysis
initial_df = pd.read_csv('./Primary_Dataset_1.csv', parse_dates=['Year'], index_col=['Year'])
initial_df.drop(['propAgr', 'propManu'], axis=1, inplace=True)
initial_df.head()
# Reference: [Year = 0,pop = 1,prodElectric = 2,lenRail,prodAgr,numPatents,numMiners,numCorps,gdpPerCap]
# Set Max Timelag
maxlag = 8

# Data Visualization Functions
# --------------------------
def visualize_sep(dataframe):
    fig, axes = plt.subplots(nrows=4, ncols=2, dpi=120, figsize=(10,6))
    for i, ax in enumerate(axes.flatten()):
        data = dataframe[dataframe.columns[i]]
        ax.plot(data, color='red', linewidth=1)
        # Decorations
        ax.set_title(str(dataframe.columns[i] + ' vs. Year'))
        ax.set_xlabel('Year')
        ax.xaxis.set_ticks_position('none')
        ax.yaxis.set_ticks_position('none')
        ax.spines["top"].set_alpha(0)
        ax.tick_params(labelsize=6)
    plt.tight_layout()
    plt.show()
def visualize_stack(dataframe):
    dataframe.plot()
    plt.show()
# visualize_sep(initial_df)
# visualize_stack(initial_df)

# Try the Granger Causality Test
# -------------------------------
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag=maxlag, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    return df
# print(grangers_causation_matrix(initial_df, variables = initial_df.columns))

# Try the Johansen Cointegration test
# --------------------------
def cointegration_test(dataframe, alpha=0.05):
    """Perform Johanson's Cointegration Test and Report Summary"""
    out = coint_johansen(dataframe,-1,5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Significant?  \n', '--'*20)
    for col, trace, cvt in zip(dataframe.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)

    print('\n' +  'Results suggest that we can go ahead with VAR...' + '\n')
# cointegration_test(initial_df)

# Check for stationarity with Augmented Dickey-Fuller Test
# -------------------------------------------------------
def int_adfuller_test(series, signif=0.05, name='', verbose=False): # also has level of significance issue.. revisit
    """Perform ADFuller to test for Stationarity of given series"""
    r = adfuller(series, autolag='AIC')
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']
    def adjust(val, length= 6): return str(val).ljust(length)
    # Print Summary
    print(f'    Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
    print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
    print(f' Significance Level    = {signif}')
    print(f' Test Statistic        = {output["test_statistic"]}')
    print(f' No. Lags Chosen       = {output["n_lags"]}')

    for key,val in r[4].items():
        print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
        print(f" => Series is Stationary.")
    else:
        print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
        print(f" => Series is Non-Stationary.")
def adfuller_test(dataframe):
    for name, column in dataframe.iteritems():
        int_adfuller_test(column, name=column.name)
        print('\n')
# adfuller_test(initial_df)

# Normalize Data
# ----------------
normalized_df = (initial_df - initial_df.mean())/ initial_df.std()
# visualize_sep(normalized_df)
# visualize_stack(normalized_df)

# Make Data Stationary by Differencing (Twice)
# --------------------------------------------
first_difference_df = normalized_df.diff().dropna()
second_difference_df = first_difference_df.diff().dropna()
# adfuller_test(second_difference_df)

# Visualize Corrected Data
# -------------------------
# visualize_sep(second_difference_df)
# visualize_stack(initial_df)
# visualize_stack(second_difference_df)
# cointegration_test(second_difference_df)

# Test out VAR model, Print Summary
# --------------------------------------
model = VAR(second_difference_df)
results = model.fit(maxlags=maxlag, ic='aic')

# Perform Durbin-Watson Test
# ----------------------------
def durbin_watson_test():
    print('Durbin-Watson Results: ')
    dw_stats = durbin_watson(results.resid)
    dw_vars = ['pop', 'prodElectric', 'lenRail', 'prodAgr', 'numPatents', 'numMiners', 'numCorps', 'gdpPerCap']
    print(dict(zip(dw_vars, dw_stats)))
# durbin_watson_test()


# Scan Results for Significant P-Values, List Results
# -----------------------------------------------------
def scan_for_significance(level_of_signif_p):
    p_val_df = pd.DataFrame(results.pvalues.gdpPerCap)
    p_val_df['Coefficients'] = results.params.gdpPerCap
    count1 = 0
    param_list = []
    print('Scanning for significant variables ...')
    for i in range(len(p_val_df)):
        if p_val_df.gdpPerCap.values[i] < level_of_signif_p:
            print('P-Val = ' + str(round(p_val_df.gdpPerCap.values[i], 4)) + ' for ' + str(p_val_df.index.values[i]) + '. Coeff = ' + str(round(results.params.gdpPerCap[i], 4)))
            count1 += 1
    print('\n' + 'Total Significant = ' + str(count1))
    return [p_val_df, param_list, count1]
# scan_for_significance(level_of_signif_p)


# Forecast VAR Model N-Years into future
# ---------------------------------------
def forecast_future(num_years, showGraph=True):
    years_to_forecast = num_years
    lag_order = results.k_ar
    forecast_input = second_difference_df.values[-lag_order:]
    fc = results.forecast(y=forecast_input, steps=years_to_forecast)
    df_forecast = pd.DataFrame(fc, index=second_difference_df.index[-years_to_forecast:], columns=second_difference_df.columns + '_2d')

    # Invert Forecast Transformations
    # --------------------------------
    def invert_transformation(initial_df, df_forecast, second_diff=False):
        """Revert back the differencing to get the forecast to original scale."""
        df_fc = df_forecast.copy()
        columns = initial_df.columns
        for col in columns:
            # Roll back 2nd Diff
            if second_diff:
                df_fc[str(col)+'_1d'] = (initial_df[col].iloc[-1]-initial_df[col].iloc[-2]) + df_fc[str(col)+'_2d'].cumsum()
            # Roll back 1st Diff
            df_fc[str(col)+'_forecast'] = initial_df[col].iloc[-1] + df_fc[str(col)+'_1d'].cumsum()
        return df_fc
    df_results = invert_transformation(initial_df, df_forecast, second_diff=True)

    if showGraph == True:
        # Plot Forecast Vs. Actual
        # --------------------------
        fig, axes = plt.subplots(nrows=int(len(initial_df.columns)/2), ncols=2, dpi=150, figsize=(10,10))
        for i, (col,ax) in enumerate(zip(initial_df.columns, axes.flatten())):
            df_results[col+'_forecast'].plot(legend=True, ax=ax).autoscale(axis='x',tight=True)
            initial_df[col][-years_to_forecast:].plot(legend=True, ax=ax);
            ax.set_title(col + ": Forecast vs Actuals")
            ax.spines["top"].set_alpha(0)
            ax.tick_params(labelsize=6)
            ax.set_ylim(0, initial_df[col][100]*2)
        plt.tight_layout()
        plt.show()
#forecast_future(10)

# Run to Generate Parameters
# scan_for_significance()
#print(scan_for_significance()[0])
# visualize_sep(initial_df)

# Genearte Econometric Equation from Parameters
def generate_equation_from_signifs(level_of_signif_p):
    p_val_df = pd.DataFrame(results.pvalues.gdpPerCap)
    p_val_df['Coefficients'] = results.params.gdpPerCap
    num_vars = 0
    param_list, coefficient_list, econometric_EQ= [],[],[]
    final_equation = ''

    for i in range(len(p_val_df)):
        if p_val_df.gdpPerCap.values[i] < level_of_signif_p:
            num_vars += 1
            coefficient_list.append(round(results.params.gdpPerCap[i], 4))
            param_list.append(p_val_df.index.values[i])
    print('\n' + 'Total Significant Vars =  ' + str(num_vars) + '\n')

    # Generate EQ from these arrays, display properly for use
    econometric_EQ_array = list(zip(coefficient_list, param_list))

    for pair in econometric_EQ_array:
        final_equation += f'{pair[0]}*{pair[1]} + '

    final_equation = ''.join((f'Equation ({level_of_signif_p} lvl) = ', final_equation, 'e'))
    print(final_equation)

#generate_equation_from_signifs(0.1)

# Do Impulse-Response Analysis
def impulse_response_analysis(variable, years_to_analyze, showGraphs = True):
    irf = results.irf(years_to_analyze)
    irfs_for_period_N = irf.irfs[years_to_analyze]
    cum_effs_for_period_N = irf.cum_effects[years_to_analyze]
    print(f'Impluse Response Functions of {variable} on GDP Per Capita for {years_to_analyze} Periods: \n')
    pprint(pd.DataFrame(irfs_for_period_N))
    print(f' \n Cumulative Effects of {variable} on GDP Per Capita for {years_to_analyze} Periods: \n')
    pprint(pd.DataFrame(cum_effs_for_period_N))
    if showGraphs == True:
        if variable == 'All':
            fig1 = irf.plot(response = 'gdpPerCap')
            fig1.tight_layout()
            plt.title('Effect of all variable impulses on GDP Per Capita')
            fig2 = irf.plot_cum_effects(response = 'gdpPerCap')
            fig2.tight_layout()

        else:
            fig1 = irf.plot(impulse = variable, response = 'gdpPerCap')
            fig1.tight_layout()
            plt.title(f'Effect of {variable} on GDP Per Capita')
            fig2 = irf.plot_cum_effects(impulse = variable, response = 'gdpPerCap')
            fig2.tight_layout()
    # insert background color parameters
        plt.show()

#    def plot(self, orth=False, *, impulse=None, response=None,
#              signif=0.05, plot_params=None, figsize=(10, 10), <-- CHANGE SIGNIF LEVEL!!!!
#              subplot_params=None, plot_stderr=True, stderr_type='asym',
#              repl=1000, seed=None, component=None):

impulse_response_analysis('All', 10, showGraphs = True)
#impulse_response_analysis('All', 10, showGraphs = True)
#pprint(pd.DataFrame(results.ma_rep(10)[10]))
