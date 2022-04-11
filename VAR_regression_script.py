# Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import warnings
import statsmodels.api as sm
from pprint import pprint
import matplotlib.gridspec as gridspec

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
raw_df = pd.read_csv('./Primary_Dataset_1.csv', parse_dates=['Year'], index_col=['Year'])
raw_df.drop(['propAgr'], axis=1, inplace=True)
#print(initial_df.head())

# Set Parameters
maxlag = None
signif_level = 0.05
start_year = 1903
end_year = 2003

# Decide Period to Analyze
# -----------------------
def set_period(start_year, end_year):
    print(f'Studying Range: {start_year} -> {end_year} ...')
    if start_year > end_year:
        print('Invalid Time Period Selected... Stopping')

    start_index = start_year - 1903
    end_index = end_year - 1903 + 1
    year_span = end_index - start_index

    initial_df = raw_df.reset_index()
    initial_df = initial_df[start_index : end_index]
    return initial_df, year_span
initial_df, year_span = set_period(start_year, end_year)
initial_df = initial_df.set_index(['Year']) # reset index back


# Edit Maxlag to work in functions requiring int val
# ---------------------------------------------------
if type(maxlag) == None:
    maxlag_temp = round(12*((year_span)/100)**(1/4)) # fundamental assumption (2003 paper)
else:
    maxlag_temp = maxlag


# Generate list of variables
# --------------------------
variable_list = []
for col in initial_df.columns:
    variable_list.append(col)
# Var List = ['pop', 'prodElectric', 'lenRail', 'prodAgr', 'numPatents', 'numMiners', 'numCorps', 'gdpPerCap']
        # Pos  0         1               2           3           4           5           6           7                (8 total)

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


# Try the Granger Causality Test (uses temp maxlag)
# -------------------------------
def grangers_causation_matrix(data, variables, test='ssr_chi2test', verbose=False):
    df = pd.DataFrame(np.zeros((len(variables), len(variables))), columns=variables, index=variables)
    for c in df.columns:
        for r in df.index:
            test_result = grangercausalitytests(data[[r, c]], maxlag = maxlag_temp, verbose=False)
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag_temp)]
            if verbose: print(f'Y = {r}, X = {c}, P Values = {p_values}')
            min_p_value = np.min(p_values)
            df.loc[r, c] = min_p_value
    df.columns = [var + '_x' for var in variables]
    df.index = [var + '_y' for var in variables]
    print(f'Maxlag used (temp) = {maxlag_temp}')
    return df
#print(grangers_causation_matrix(initial_df, variables = initial_df.columns))


# Try the Johansen Cointegration test
# --------------------------
def cointegration_test(dataframe, alpha = 0.05): # CANNOT EDIT
    out = coint_johansen(dataframe, -1, 5)
    d = {'0.90':0, '0.95':1, '0.99':2}
    traces = out.lr1
    cvts = out.cvt[:, d[str(1-alpha)]]
    def adjust(val, length= 6): return str(val).ljust(length)

    # Summary
    print('Name   ::  Test Stat > C(95%)    =>   Significant?  \n', '--'*20)
    for col, trace, cvt in zip(dataframe.columns, traces, cvts):
        print(adjust(col), ':: ', adjust(round(trace,2), 9), ">", adjust(cvt, 8), ' =>  ' , trace > cvt)
#cointegration_test(initial_df)


# Check for stationarity with Augmented Dickey-Fuller Test (uses temp maxlag)
# -------------------------------------------------------
def int_adfuller_test(series, signif = signif_level, name = '', verbose=False, showGraph = False):
    r = adfuller(series, autolag=None, maxlag = maxlag_temp)
    output = {'test_statistic':round(r[0], 4), 'pvalue':round(r[1], 4), 'n_lags':round(r[2], 4), 'n_obs':r[3]}
    p_value = output['pvalue']
    def adjust(val, length= 6): return str(val).ljust(length)
    # Print Summary if True
    if showGraph == True:
        print(f' Augmented Dickey-Fuller Test on "{name}"', "\n   ", '-'*47)
        print(f' Null Hypothesis: Data has unit root. Non-Stationary.')
        print(f' Significance Level    = {signif}')
        print(f' Test Statistic        = {output["test_statistic"]}')
        print(f' No. Lags Chosen       = {output["n_lags"]}')

        for key,val in r[4].items():
            print(f' Critical value {adjust(key)} = {round(val, 3)}')

    if p_value <= signif:
        if showGraph == True:
            print(f" => P-Value = {p_value}. Rejecting Null Hypothesis.")
            print(f" => Series is Stationary. \n")
        status = 1
    else:
        if showGraph == True:
            print(f" => P-Value = {p_value}. Weak evidence to reject the Null Hypothesis.")
            print(f" => Series is Non-Stationary. \n")
        status = 0
    return status
def adfuller_test(dataframe, showgraph = False):
    stationary_index_list = []
    for name, column in dataframe.iteritems():
        result = int_adfuller_test(column, name=column.name, showGraph = showgraph)
        stationary_index_list.append(result)
    return stationary_index_list
#adfuller_test(initial_df, showgraph = True)
#print(adfuller_test(initial_df))


# Normalize Data
# ----------------
normalized_df = (initial_df - initial_df.mean())/ initial_df.std()


# Function to Make Data Stationary by Differencing (until adfuller = all True)
# -----------------------------------------------------------------------
def make_stationary(dataframe, showCount = False):
    difference_count = 0
    while sum(adfuller_test(dataframe)) < len(dataframe.columns):
        dataframe = dataframe.diff().dropna()
        difference_count += 1
    if showCount == True:
        print(f'Making Stationary...')
        print(f'Differenced {difference_count} time(s), all vars now stationary \n')
    print(f'Difference Count = {difference_count}')
    return dataframe
final_df = make_stationary(normalized_df)
#visualize_sep(final_df)


# Visualize Final Data
# -------------------------
#visualize_sep(final_df)
#visualize_stack(final_df)


# Test out VAR model, Print Summary
# --------------------------------------
model = VAR(final_df)
results = model.fit(maxlags = maxlag, ic='aic')
print(f'Model uses Lag Order: {results.k_ar} and Significance Lvl: {round((1-signif_level)*100)}% \n')
#print(results.summary())


# Perform Durbin-Watson Test
# ----------------------------
def durbin_watson_test():
    print('Durbin-Watson Results: ')
    dw_stats = durbin_watson(results.resid)
    dw_vars = variable_list
    print(dict(zip(dw_vars, dw_stats)))
#durbin_watson_test()


# Scan Results for Significant P-Values, List Results
# -----------------------------------------------------
def scan_for_significance(level_of_signif_p = signif_level):
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
#scan_for_significance()


# Forecast VAR Model N-Years into future -- MODIFY TO GIVE % ERROR (Avg Diff?)
# ---------------------------------------
def forecast_future(num_years, showGraph=True):
    years_to_forecast = num_years
    lag_order = results.k_ar
    forecast_input = final_df.values[-lag_order:]
    fc = results.forecast(y=forecast_input, steps=years_to_forecast)
    df_forecast = pd.DataFrame(fc, index=final_df.index[-years_to_forecast:], columns=final_df.columns + '_2d')

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
forecast_future(num_years = 10)


# Generate Econometric Equation from Parameters
# --------------------------------------------
def generate_equation_from_signifs(level_of_signif_p = signif_level):
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
    print('\n')
generate_equation_from_signifs()


# Do Impulse-Response Analysis
# -------------------------------
def impulse_response_analysis(variable, years_to_analyze, signif_level = signif_level, showGraphs = False):
    irf = results.irf(years_to_analyze)
    irfs_for_period_N = irf.irfs[years_to_analyze]
    cum_effs_for_period_N = irf.cum_effects[years_to_analyze]

    # Create Table of IRF's
    df_irf = pd.DataFrame(data = irfs_for_period_N, index = variable_list, columns = variable_list)

    # Create Table of CumEff's
    df_cumEff = pd.DataFrame(data = cum_effs_for_period_N, index = variable_list, columns = variable_list)

    # Handle 'All' or variable print
    #if variable == 'All':
        #print(f'Impluse Response Functions of All Variables for {years_to_analyze} Periods: \n')
        #pprint(df_irf)
        # print(f'\n\nCumulative Effects of All Variables for {years_to_analyze} Periods: \n')
        # pprint(df_cumEff)
    #else:
        #print(f'\nImpluse Response Functions of {variable} on GDP Per Capita for {years_to_analyze} Periods: \n')
        #pprint(df_irf[variable])
    # Show Graphs
    if showGraphs == True:
        if variable == 'All':
            irf_plot = irf.plot(response = 'gdpPerCap', signif = signif_level)
            #cum_plot = irf.plot_cum_effects(response = 'gdpPerCap', signif = signif_level)
            plt.tight_layout()
        else:
            fig1 = irf.plot(impulse = variable, response = 'gdpPerCap', signif = signif_level)
            #fig2 = irf.plot_cum_effects(impulse = variable, response = 'gdpPerCap', signif = signif_level)
            plt.tight_layout()
        plt.show()
impulse_response_analysis('All', 10, showGraphs = True)
#impulse_response_analysis('prodElectric', 10, showGraphs = True)


# Do Simple OLS on Data Using Given Vars Linearly -- EXP WITH NONLIN FORMS
# ---------------------------------------------
def linear_OLS(dataframe):
    X = dataframe[['pop', 'prodElectric', 'lenRail', 'prodAgr', 'numPatents', 'numMiners', 'numCorps']]
    y = dataframe['gdpPerCap']
    X = sm.add_constant(X)
    est = sm.OLS(y, X).fit()
    print(est.summary())
#linear_OLS(initial_df)


# Conduct Time-Series VAR Analysis (Directory) -- COME BACK TO THIS
# ------------------------------------------
#def time_series_VAR():



























# placemarker
