import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, kpss


mu, sigma = 1, 0.1
x = np.random.normal(mu, sigma, 1000)


plt.plot(x)
plt.grid()
plt.show()


fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)
fig = plot_acf(x, lags=20, ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(x, lags=20, ax=ax2)
fig.show()


#Adfuller
#https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
#https://machinelearningmastery.com/time-series-data-stationary-python/
result = adfuller(x)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print('Critical Values:')
for key, value in result[4].items():
	print('\t%s: %.3f' % (key, value))
print(f'Adfuller Result: The series is {"not " if result[1] >= 0.05 else ""}stationary')
print("")


#KPSS
#https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html
statistic, p_value, n_lags, critical_values = kpss(x)
print(f'KPSS Statistic: {statistic}')
print(f'p-value: {p_value}')
print(f'num lags: {n_lags}')
print('Critial Values:')
for key, value in critical_values.items():
    print(f'   {key} : {value}')

print(f'KPSS Result: The series is {"not " if p_value < 0.05 else ""}stationary')

