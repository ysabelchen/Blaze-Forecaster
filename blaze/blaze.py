# Optimizing modules
from __future__ import division
from scipy.optimize import minimize
from sklearn.model_selection import TimeSeriesSplit

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import itertools
import warnings
import os
import datetime as dt

# TSA from Statsmodels
import statsmodels.api as sm
import statsmodels.formula.api as smf
import statsmodels.tsa.api as smt
from statsmodels.tsa.api import ExponentialSmoothing, SimpleExpSmoothing, Holt
import scipy.stats as st
import statsmodels.stats.api as sms
#!pip install --pre statsmodels --upgrade #!pip install tensorflow #!pip install keras
#!pip install cython #!pip install pystan #!pip install fbprophet

#Prophet Package
from prophet import Prophet
from prophet.diagnostics import cross_validation
from prophet.diagnostics import performance_metrics

warnings.filterwarnings('ignore')
# %matplotlib inline

pd.set_option('display.float_format', lambda x: '%.5f' % x) # pandas
np.set_printoptions(precision=5, suppress=True) # numpy

pd.set_option('display.max_columns', 100)
pd.set_option('display.max_rows', 100)

# seaborn plotting style
sns.set(style='ticks', context='poster')

class Blaze:
    file = '/TimeSeriesData.csv'
    path = os.getcwd()

    data = pd.read_csv('TimeSeriesData.csv', parse_dates=[0])
    data = data.dropna()
    data.head()

    data['timestamp'] = data['timestamp'].dt.strftime("%Y-%m-%d")
    data['timestamp'] = data['timestamp'].apply(lambda x: pd.to_datetime(x))
    data = data.drop_duplicates(['timestamp','Variable1','Variable2', 'Variable3', 'target'])
    data = data.groupby(by=['timestamp','Variable1','Variable2', 'Variable3'], axis=0, as_index=False).max()
    data.head()

    data = data.set_index(pd.DatetimeIndex(data['timestamp']),drop=True)
    data.head()

    data2 = data.drop(['timestamp'],axis=1)
    data2.head


    ### Helper functions to prepare and clean the data ###
    # Allows picking certain variable combinations for creating a model
    def data_parser(data,Variable1_type,Variable2_type,Variable3_type):
        data = data[(data['Variable1'] == str(Variable1_type)) &
                    (data['Variable2'] == str(Variable2_type)) &
                    (data['Variable3'] == str(Variable3_type))]
        return data.drop(['Variable1','Variable2','Variable3'],axis=1)
    
    # Generates a date range for the input data set and checks to ensure that the data does not have any date gaps
    # Parameters:
        # data: DataFrame containing the dataset
    # Returns: DataFrame with the missing data gaps, extact months and overall count/month. A distribution of the missing datapoints by month
    def find_gaps(data):
        start_date = str(data.index[0]).split(' ')[0]
        end_date = str(data.index[-1]).split(' ')[0]
        date_range = pd.date_range(start=start_date, end=end_date, freq='D')
        date_range = pd.DataFrame(date_range,columns=['timestamp']).set_index('timestamp')
        df = date_range.join(data)
        missing = df[df['target'].isnull()]
        missing_stats = missing.reset_index()
        missing_stats = missing_stats.groupby(missing_stats['timestamp'].dt.strftime('%B-%Y'))\
                        ['target'].size()
        print('The data contains: {} missing days'.
            format(df['target'].isnull().value_counts()[1]))
        print('\n')
        print('Summary of missing days by month and year: %s' %missing_stats)
        print(missing_stats.plot(kind='bar'),plt.title('Distribution of Missing months',fontsize=14))
        return df, missing
    

    ### Load the data ###
    data_parsed = data_parser(data2,'US2','HDD', 'Commercial')
    data_parsed.head()
    data_cleaned, missing = find_gaps(data_parsed)
    _ = data_cleaned.plot(figsize=(12,8), title="Target Variable Overtime")


    ### Imputation Strategy ###
    modeling_data = data_cleaned 
    modeling_data = modeling_data.fillna(modeling_data.bfill())
    modeling_data.plot(figsize=(12,8), title="Target Variable Overtime")


    ### Holdouts: Split data into train and test sets ###
    n_sample = len(modeling_data)
    n_train=int(0.85*n_sample)+1 # Use 85% of the data for training
    n_forecast=n_sample-n_train
    # Use iloc to split the data by index location
    ts_train = modeling_data.iloc[:n_train]['target']
    ts_test = modeling_data.iloc[n_train:]['target']
    print("Training Count:", ts_train.shape)
    print("Testing Count:", ts_test.shape, "\n")
    print("Training Series:", "\n", ts_train.tail(), "\n")
    print("Testing Series:", "\n", ts_test.head())


    ### Plot the data ###
    ts_train.plot(figsize=(15,8), title= 'Target Overtime', fontsize=14, label='Training Data')
    ts_test.plot(figsize=(15,8), title= 'Target Overtime', fontsize=14, label='Test Data')
    plt.xlabel('Date')
    plt.ylabel('Target')
    plt.legend(loc='best')
    plt.show()


    ### Visualize training set ###
    # Examines the patterns of ACF and PACF, along with the time series plot and histogram
    def tsplot(y, lags=None, title='', figsize=(14, 8)):
        fig = plt.figure(figsize=figsize)
        layout = (2, 2)
        ts_ax   = plt.subplot2grid(layout, (0, 0))
        hist_ax = plt.subplot2grid(layout, (0, 1))
        acf_ax  = plt.subplot2grid(layout, (1, 0))
        pacf_ax = plt.subplot2grid(layout, (1, 1))
        
        y.plot(ax=ts_ax)
        ts_ax.set_title(title)
        y.plot(ax=hist_ax, kind='hist', bins=25)
        hist_ax.set_title('Histogram')
        smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
        smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
        [ax.set_xlim(0) for ax in [acf_ax, pacf_ax]]
        sns.despine()
        fig.tight_layout()
        return ts_ax, acf_ax, pacf_ax
    
    tsplot(ts_train, title='A Given Training Series', lags=35)


    ### Performance Metrics ###
    # Defining the accuracy metrics
    from sklearn.metrics import r2_score, median_absolute_error, mean_absolute_error
    from sklearn.metrics import median_absolute_error, mean_squared_error, mean_squared_log_error

    def get_rmse(y, y_hat):
        mse = np.mean((y - y_hat)**2)
        return np.sqrt(mse)

    def get_mape(y, y_hat):
        perc_err = (100*(y - y_hat))/y
        return np.mean(abs(perc_err))

    def get_mase(y, y_hat):
        abs_err = abs(y - y_hat)
        dsum=sum(abs(y[1:] - y_hat[1:]))
        t = len(y)
        denom = (1/(t - 1))* dsum
        return np.mean(abs_err/denom)
    

    ### Method 1: Naive Approach ###
    dd= np.asarray(ts_train)
    y_hat = ts_test.copy()
    y_hat[:] = dd[len(dd)-1]
    plt.figure(figsize=(12,8))
    plt.plot(ts_train.index, ts_train, label='Training Data')
    plt.plot(ts_test.index, ts_test, label='Test Data')
    plt.plot(ts_test.index, y_hat, label='Naive Prediction')
    plt.legend(loc='best')
    plt.xlabel('Date')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    plt.ylabel('Target')
    plt.title("Naive Model")
    plt.show()

    rmse_naive = get_rmse(ts_test, y_hat)
    print("RMSE Naive Test: ", rmse_naive)

    mape_naive = get_mape(ts_test, y_hat)
    print("MAPE Naive Test: ", mape_naive)

    mase_naive = get_mase(ts_test, y_hat)
    print("MASE Naive Test: ", mase_naive)


    ### Method 2: Simple Average ###
    y_hat_avg = ts_test.copy()
    y_hat_avg[:] = ts_train.mean()
    plt.figure(figsize=(12,8))
    plt.plot(ts_train.index, ts_train, label='Training Data')
    plt.plot(ts_test.index, ts_test, label='Test Data')
    plt.plot(ts_test.index, y_hat_avg, label='Average Prediction')
    plt.legend(loc='best')
    plt.xlabel('Date')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    plt.ylabel('Target')
    plt.title("Average Model")
    plt.show()

    rmse_average = get_rmse(ts_test, y_hat_avg)
    print("RMSE Average Test: ", rmse_average)

    mape_average = get_mape(ts_test, y_hat_avg)
    print("MAPE Average Test: ", mape_average)

    mase_average = get_mase(ts_test, y_hat_avg)
    print("MASE Average Test: ", mase_average)


    ### Method 3: Moving Average ###
    y_hat_avg = ts_test.copy()
    y_hat_avg[:] = ts_train.rolling(30).mean().iloc[-1]
    plt.figure(figsize=(12,8))
    plt.plot(ts_train.index, ts_train, label='Training Data')
    plt.plot(ts_test.index, ts_test, label='Test Data')
    plt.plot(ts_test.index, y_hat_avg, label='Moving Average Prediction')
    plt.legend(loc='best')
    plt.xlabel('Date')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    plt.ylabel('Target')
    plt.title("Moving Average Model")
    plt.show()

    rmse_moving_average = get_rmse(ts_test, y_hat_avg)
    print("RMSE Moving Average Test: ", rmse_moving_average)

    mape_moving_average = get_mape(ts_test, y_hat_avg)
    print("MAPE Moving Average Test: ", mape_moving_average)

    mase_moving_average = get_mase(ts_test, y_hat_avg)
    print("MASE Moving Average Test: ", mase_moving_average)


    ### Method 4: Simple Exponential Smoothing ###
    # Simple exponential smoothing (Single exponential smoothing)
    y_hat_avg = ts_test.copy()
    Optimum_Parameter_Fit1 = SimpleExpSmoothing(np.asarray(ts_train)).fit(optimized=True)
    alpha_simple_exp= Optimum_Parameter_Fit1.model.params['smoothing_level']
    fit1 = SimpleExpSmoothing(np.asarray(ts_train)).fit(smoothing_level=alpha_simple_exp, optimized=True)
    y_hat_avg[:] = fit1.forecast(len(ts_test))
    plt.figure(figsize=(12,8))
    plt.plot(ts_train.index, ts_train, label='Training Data')
    plt.plot(ts_test.index, ts_test, label='Test Data')
    plt.plot(ts_test.index, y_hat_avg, label='Exponential Smoothing Prediction')
    plt.legend(loc='best')
    plt.xlabel('Date')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    plt.ylabel('Target')
    plt.title("Exponential Smoothing Model")
    plt.show()

    rmse_exponential_smothing = get_rmse(ts_test, y_hat_avg)
    print("RMSE Exponential Smothing Test: ", rmse_exponential_smothing)

    mape_exponential_smothing = get_mape(ts_test, y_hat_avg)
    print("MAPE Exponential Smothing Test: ", mape_exponential_smothing)

    mase_exponential_smothing = get_mase(ts_test, y_hat_avg)
    print("MASE Exponential Smothing Test: ", mase_exponential_smothing)


    ### Method 5: Holt's Linear Trend ###
    y_hat_avg = ts_test.copy()
    Optimum_Parameter_Fit2 = Holt(np.asarray(ts_train)).fit(optimized=True)
    alpha_holt= Optimum_Parameter_Fit2.model.params['smoothing_level']
    beta_holt= Optimum_Parameter_Fit2.model.params['smoothing_trend'] #edit: prev: smoothing_slope
    fit2 = Holt(np.asarray(ts_train)).fit(smoothing_level=alpha_holt, smoothing_trend=beta_holt, 
                                        optimized=True, damping_slope=0.95) #edit: prev: smoothing_slope
    y_hat_avg[:] = fit2.forecast(len(ts_test))
    plt.figure(figsize=(12,8))
    plt.plot(ts_train.index, ts_train, label='Training Data')
    plt.plot(ts_test.index, ts_test, label='Test Data')
    plt.plot(ts_test.index, y_hat_avg, label='Holt Linear Trend Prediction')
    plt.legend(loc='best')
    plt.xlabel('Date')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    plt.ylabel('Target')
    plt.title("Holt Linear Trend Model")
    plt.show()

    rmse_Holt_Linear = get_rmse(ts_test, y_hat_avg)
    print("RMSE Holt Linear Trend Test: ", rmse_Holt_Linear)

    mape_Holt_Linear = get_mape(ts_test, y_hat_avg)
    print("MAPE Holt Linear Trend Test: ", mape_Holt_Linear)

    mase_Holt_Linear = get_mase(ts_test, y_hat_avg)
    print("MASE Holt Linear Trend Test: ", mase_Holt_Linear)


    ### Method 6: Holt-Winters Seasonal ###
    # Parameters:
        # params: Vector of parameters for optimization
        # series: Dataset with timeseries
    # Returns: Error on CV
    def timeseriesCVscore(params, series, loss_function=mean_squared_error):
        errors = []
        
        values = series.target
        alpha, beta, gamma = params
        
        # set the number of folds for cross-validation
        tscv = TimeSeriesSplit(n_splits=3) 
        
        # iterating over folds, train model on each, forecast and calculate error
        # The additive method is preferred when the seasonal variations are roughly constant through the series, 
        # while the multiplicative method is preferred when the seasonal variations are changing proportional to the level of the series.
        for train, test in tscv.split(values):

            model = ExponentialSmoothing(values[train], seasonal_periods=30 ,trend='additive', 
                                        seasonal='additive',).fit(smoothing_level=alpha,smoothing_slope = beta,
                                                                smoothing_seasonal = gamma, damping_slope=0.95) 
            predictions = model.forecast(len(test))
            actual = values[test]
            error = loss_function(predictions, actual)
            errors.append(error)
            
        return np.mean(np.array(errors))
    
    data_cv =pd.DataFrame(ts_train).reset_index()
    x = [0, 0, 0] 
    opt = minimize(timeseriesCVscore, x0=x, 
                args=(data_cv, mean_squared_log_error), 
                method="Nelder-Mead", bounds = ((0, 1), (0, 1), (0, 1)))

    alpha_final, beta_final, gamma_final = opt.x
    # print("Optimum Alpha = ", alpha_final, "Optimum Beta = ", beta_final, "Optimum Gamma = ", gamma_final)

    ''' Applying exponential smoothing to the seasonal components in addition to level and trend
        The additive model is useful when the seasonal variation is relatively constant over time which is this case. 
        Additive:  x = Trend + Seasonal + Random
        The multiplicative model is useful when the seasonal variation increases over time.
        Multiplicative:  x = Trend * Seasonal * Random
        smoothing_level – The alpha value of the simple exponential smoothing.
        smoothing_slope – The beta value of the Holt’s trend method.
        smoothing_seasonal – The gamma value of the holt winters seasonal method.
        seasonal_periods: season length for Holt-Winters model
        Need to find the optimum parameters for level, trend and seasonal '''
    y_hat_avg = ts_test.copy()
    fit3 = ExponentialSmoothing(np.asarray(ts_train), seasonal_periods=30,trend='additive', 
                                seasonal='additive',).fit(smoothing_level=alpha_final,smoothing_slope = beta_final, 
                                                    smoothing_seasonal = gamma_final, optimized=True, damping_slope=0.95)
    y_hat_avg[:] = fit3.forecast(len(ts_test))
    plt.figure(figsize=(12,8))
    plt.plot(ts_train.index, ts_train, label='Training Data')
    plt.plot(ts_test.index, ts_test, label='Test Data')
    plt.plot(ts_test.index, y_hat_avg, label='Holt-Winter Prediction')
    plt.legend(loc='best')
    plt.xlabel('Date')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    plt.ylabel('Target')
    plt.title("Holt-Winter Model")
    plt.show()

    # Holt-Winters Seasonal Model Performance
    rmse_Holt_Winter = get_rmse(ts_test, y_hat_avg)
    print("RMSE Holt Winter Seasonal Test: ", rmse_Holt_Winter)

    mape_Holt_Winter = get_mape(ts_test, y_hat_avg)
    print("MAPE Holt Winter Seasonal Test: ", mape_Holt_Winter)

    mase_Holt_Winter = get_mase(ts_test, y_hat_avg)
    print("MASE Holt Winter Seasonal Test: ", mase_Holt_Winter)


    ### Method 7: SARIMA ###
    p = d = q = range(0, 2)
    # Generate all different combinations of p, q and q triplets
    pdq = list(itertools.product(p, d, q))
    # Generate all different combinations of seasonal p, q and q triplets
    seasonal_pdq = [(x[0], x[1], x[2], 30) for x in list(itertools.product(p, d, q))]

    # Grid search to find the optimum parameters for the SARIMA model
    def model_3(train_data,i,j):
        min_AIC_param = []
        mod = sm.tsa.statespace.SARIMAX(train_data,
                                            order=i,
                                            seasonal_order=j,
                                            enforce_stationarity=False,
                                            enforce_invertibility=False)
        results = mod.fit()
        min_AIC_param.append((results.aic,i,j))
        return min_AIC_param
    
    optimal_params = min((model_3(ts_train, i, j) for i in pdq for j in seasonal_pdq))
    
    print("The optimum SARIMA AIC is: {}, pdq: {}, seasonal_pdq is: {}" .format(optimal_params[0][0], # AIC score
                                                                     optimal_params[0][1], # p, d, q
                                                                     optimal_params[0][2]))# seasonal p, d, q
    
    # SARIMA Model Estimation
    mod = sm.tsa.statespace.SARIMAX(modeling_data,
                                order=optimal_params[0][1], # normal p, d, q parameters
                                seasonal_order=optimal_params[0][2], # seasonal p, d, q parameters
                                enforce_stationarity=False,
                                enforce_invertibility=False)

    results = mod.fit()
    results.plot_diagnostics(figsize=(15, 12))
    plt.show()

    pred_dynamic = results.get_prediction(start=n_train, dynamic=True, full_results=True)
    pred_dynamic_ci = pred_dynamic.conf_int(alpha = .1)
    pred_mean = pred_dynamic.predicted_mean

    y_hat_avg = ts_test.copy()
    pred_begin = ts_test.index[0]
    pred_end = ts_test.index[-1]
    y_hat_avg[:] = results.predict(start=pred_begin.strftime('%Y-%m-%d'), end=pred_end.strftime('%Y-%m-%d'), dynamic=True)
    plt.figure(figsize=(12,8))
    plt.plot(ts_train.index, ts_train, label='Training Data')
    plt.plot(ts_test.index, ts_test, label='Test Data')
    plt.plot(ts_test.index, y_hat_avg, label='SARIMA Prediction')
    plt.legend(loc='best')
    plt.xlabel('Date')
    locs, labels = plt.xticks()
    plt.setp(labels, rotation=45)
    plt.ylabel('Target')
    plt.title("SARIMA Model")

    # SARIMA Model plot with Confidence Interval
    ax = modeling_data[0:].plot(label='Observed', figsize=(12, 8))
    pred_dynamic.predicted_mean.plot(label='SARIMA Prediction', ax=ax)
    ax.fill_between(pred_dynamic_ci.index,
                    pred_dynamic_ci.iloc[:, 0],
                    pred_dynamic_ci.iloc[:, 1], color='k', alpha=.25)
    ax.fill_betweenx(ax.get_ylim(), modeling_data.index[n_train], modeling_data.index[-1],
                    alpha=.1, zorder=-1)
    ax.set_title('SARIMA Model with Confidence Interval')
    ax.set_xlabel('Date')
    ax.set_ylabel('Target')

    plt.legend()
    plt.show()

    # SARIMA Model Forecasting
    pred_uc = results.get_forecast(steps=90)
    pred_ci = pred_uc.conf_int(alpha = .1)

    ax = modeling_data[0:].plot(label='observed', figsize=(12,8))
    pred_uc.predicted_mean.plot(ax=ax, label='SARIMA Model Forecast')
    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=.25)
    ax.set_xlabel('Date')
    ax.set_ylabel('Target')

    plt.title("Target Model Forecast")
    plt.legend()
    plt.show()

    pred_forecast=pred_uc.predicted_mean.to_frame(name='Forecast')
    SARIMA_Forecast=pd.concat([pred_forecast, pred_ci], axis=1, ignore_index=True)
    SARIMA_Forecast.columns=['SARIMA Forecast', 'SARIMA Lower Limit', 'SARIMA Upper Limit']

    # Table that shows Forecast with Confidence Interval for future dates
    SARIMA_Forecast

    # SARIMA Model Performance
    rmse_SARIMA = get_rmse(ts_test, y_hat_avg)
    print("RMSE SARIMA Test: ", rmse_SARIMA)

    mape_SARIMA = get_mape(ts_test, y_hat_avg)
    print("MAPE SARIMA Test: ", mape_SARIMA)

    mase_SARIMA = get_mase(ts_test, y_hat_avg)
    print("MASE SARIMA Test: ", mase_SARIMA)


    ### Method 8: Prophet ###
    ProphetData = data_parsed.reset_index()
    ProphetData = ProphetData.rename(columns={'timestamp':'ds','target':'y'}) #edit: prev: 'index'
    ProphetTrainData = ts_train.reset_index()
    ProphetTrainData = ProphetTrainData.rename(columns={'timestamp':'ds','target':'y'})
    ProphetTestData = ts_test.reset_index()
    ProphetTestData = ProphetTestData.rename(columns={'timestamp':'ds','target':'y'})

    Prophet_Model = Prophet(interval_width=0.9, seasonality_mode='additive', weekly_seasonality=False)
    Prophet_Model.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    Prophet_Model.fit(ProphetData)

    future_dates = Prophet_Model.make_future_dataframe(periods=90)
    Prophet_forecast = Prophet_Model.predict(future_dates)

    # ds: the datestamp of the forecasted value
    # yhat: the forecasted value of our metric
    # yhat_lower: the lower bound of our forecast
    # yhat_upper: the upper bound of our forecast
    Prophet_forecast_subset = Prophet_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]

    Prophet_Model.plot(Prophet_forecast, uncertainty=True)
    
    # Prophet Model Performance
    Prophet_Model_Train = Prophet(interval_width=0.9, seasonality_mode='additive', weekly_seasonality=False)
    Prophet_Model_Train.add_seasonality(name='monthly', period=30.5, fourier_order=5)
    Prophet_Model_Train.fit(ProphetTrainData)

    Predicted_Prophet_Test_Frame=Prophet_Model_Train.make_future_dataframe(periods=len(ProphetTestData))
    Predicted_Prophet_Test_Data=Prophet_Model_Train.predict(Predicted_Prophet_Test_Frame)

    Predicted_Prophet_Test_Data=Predicted_Prophet_Test_Data[-len(ProphetTestData):]
    y_true, y_pred = np.array(ProphetTestData['y']), np.array(Predicted_Prophet_Test_Data['yhat'])
    rmse_Prophet = get_rmse(y_true, y_pred)
    print("RMSE Prophet Test: ", rmse_Prophet)

    mape_Prophet = get_mape(y_true, y_pred)
    print("MAPE Prophet Test: ", mape_Prophet)

    mase_Prophet = get_mase(y_true, y_pred)
    print("MASE Prophet Test: ", mase_Prophet)

    # Prophet Model Forecasting
    Prophet_forecast_values = Prophet_forecast_subset[-90:]
    Prophet_forecast_values_reset = Prophet_forecast_values.set_index('ds').rename_axis('Date')
    Prophet_forecast_values_reset = Prophet_forecast_values_reset.rename(columns={'yhat':'Prophet Forecast','yhat_lower':'Prophet Lower Limit','yhat_upper':'Prophet Upper Limit'})
    Prophet_forecast_values_reset


    ### Model Performance of All Models ###
    Performance = pd.DataFrame({'Naive':[rmse_naive, mape_naive, mase_naive],
                    'Simple Average':[rmse_average, mape_average, mase_average],
                    'Moving Average':[rmse_moving_average, mape_moving_average, mase_moving_average],
                    'Exponential Smothing': [rmse_exponential_smothing, mape_exponential_smothing, mase_exponential_smothing],
                    'Holt Linear Trend':[rmse_Holt_Linear, mape_Holt_Linear, mase_Holt_Linear], 
                    'Holt-Winter': [rmse_Holt_Winter, mape_Holt_Winter, mase_Holt_Winter],
                    'SARIMA': [rmse_SARIMA, mape_SARIMA, mase_SARIMA],
                    'Prophet':[rmse_Prophet, mape_Prophet, mase_Prophet],
                    'Blaze (SARIMA and Prophet)': [(rmse_SARIMA+rmse_Prophet)/2, (mape_SARIMA+mape_Prophet)/2, 
                                                       (mase_SARIMA+mase_Prophet)/2]},
                    index=['RMSE', 'MAPE', 'MASE'])
    Performance.T


    ### Blaze Model Forecasting ###

    # Build the hybrid forecast, starting with the SARIMA model. 
    Hybrid_Forecast = SARIMA_Forecast.reset_index()
    Hybrid_Forecast = Hybrid_Forecast.set_index('index').rename_axis('Date')

    # Add Prophet model using a left join
    Hybrid_Forecast_All = pd.merge(Hybrid_Forecast, Prophet_forecast_values_reset, how='left', on='Date')

    # Create the hybrid Blaze model based on the simple average of the SARIMA and Prophet forecast
    Hybrid_Forecast_All['Blaze Forecast']=(Hybrid_Forecast_All['SARIMA Forecast']/2 + Hybrid_Forecast_All['Prophet Forecast']/2)
    Hybrid_Forecast_All['Blaze Lower Limit']=(Hybrid_Forecast_All['SARIMA Lower Limit']/2 + Hybrid_Forecast_All['Prophet Lower Limit']/2)
    Hybrid_Forecast_All['Blaze Upper Limit']=(Hybrid_Forecast_All['SARIMA Upper Limit']/2 + Hybrid_Forecast_All['Prophet Upper Limit']/2)

    Hybrid_Forecast_Final=Hybrid_Forecast_All[['Blaze Forecast', 'Blaze Lower Limit', 'Blaze Upper Limit', 'SARIMA Forecast', 'Prophet Forecast' ]]
    Hybrid_Forecast_Final

    # Export the forecast of the models to an excel file
    Hybrid_Forecast_Final.to_excel('Target Forecast.xlsx', sheet_name='Forecast')