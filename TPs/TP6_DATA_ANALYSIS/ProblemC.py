import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.stattools import kpss, adfuller
import seaborn as sns

df = pd.read_csv("sales.csv")
train = df.drop(index=["2020-0" + str(i) for i in range(1,10)], axis=1)['Fuel']
test = df.iloc[48:59, :]['Fuel']

model = ARIMA(train, order=(1,0,1))
model_fit = model.fit(disp=0)


residuals = pd.DataFrame(model_fit.resid)

#Shapiro
print("Shapiro test:  statistic -> ", stats.shapiro(residuals)[0], " / pValue -> ", stats.shapiro(residuals)[1], f' \n                                / Result (alpha=0.05) : The series is {"not " if stats.shapiro(residuals)[1] <= 0.05 else ""}normally distributed')

#Boxpierce
print("BoxPierce test:  / pValue -> ", acorr_ljungbox(residuals)[1][0], f'\n                                 / Result (alpha=0.05) : The data are {"not " if acorr_ljungbox(residuals)[1][0] <= 0.05 else ""}independently distributed')

#Adfuller
#https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.adfuller.html
#https://machinelearningmastery.com/time-series-data-stationary-python/
result = adfuller(residuals)
print('ADF Statistic: %f' % result[0])
print('p-value: %f' % result[1])
print(f'Adfuller Result: The series is {"not " if result[1] >= 0.05 else ""}stationary')
print("")


#KPSS
#https://www.statsmodels.org/stable/generated/statsmodels.tsa.stattools.kpss.html
statistic, p_value, n_lags, critical_values = kpss(residuals)
print(f'KPSS Statistic: {statistic}')
print(f'p-value: {p_value}')
print(f'KPSS Result: The series is {"not " if p_value < 0.05 else ""}stationary')

sns.set_theme()

sns.distplot(residuals)
plt.title('Histogram of the residuals of ARMA(1,1)')
plt.show()
