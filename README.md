# Blaze-Forecaster
PyPI Blaze Forecaster: https://pypi.org/project/blaze-forecaster/#description
The package employs a combination of time series models: SARIMA and Prophet.
Stats models and Prophet are required installations to run the Blaze forecaster package.

The process begins in the standard form: importing and cleaning data, imputation of data, splitting of the data into training and test sets, and plotting the data. Then, autocorrelation (ACF) measures the correlation between consecutive instances, adding in the lagged factor. Partial autocorrelation (PACF) also measures the degree of association between y(t) and y(t-p) removing the effects of lags in between. The patterns of ACF and PACF are then examined to identify the types of occurring seasonalities. The final step of this standard process is defining the accuracy metrics: root mean square error (RMSE), mean absolute percent error (MAPE), and mean absolute scaled error (MASE).

The Blaze model is built using the simple average of the SARIMA and Prophet forecasts. We compared the Blaze model against eight other popular models, consisting of the Naive Approach, Simple Average, Moving Average, Exponential Smoothing, Holt Linear Trend, Holt-Winters, SARIMA, and Prophet. Blaze produces the third-lowest RMSE and MAPE after SARIMA and Holt Linear Trend, and produces a MASE < 1, indicating a better performance compared to the naive forecast.

This model includes seasonality and balances SARIMA’s use of the autoregressive model and order of integration with Prophet’s use of the additive model and resistance to dramatic changes. In doing so, the Blaze forecaster offers a new procedure for forecasting time series data that accounts for a diverse set of features.
