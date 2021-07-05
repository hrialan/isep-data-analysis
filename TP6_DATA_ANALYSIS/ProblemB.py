import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import kpss, adfuller
from statsmodels.stats.diagnostic import acorr_ljungbox


df = pd.read_csv("sales.csv")
df = df.drop(index=["2020-0" + str(i) for i in range(1,10)], axis=1)


plt.title('Food and Fuel sales over time')
plt.plot(df['Food'],label='Food sales')
plt.plot(df['Fuel'],label='Fuel sales')
plt.legend(loc="lower right", frameon=False)
plt.ylabel('Sales')
plt.xlabel('Date')
xAxis = ["01_2016"]
for i in range(46):
    xAxis.append(" ")
xAxis.append("12-2019")
plt.grid()
plt.gca().axes.xaxis.set_ticklabels(xAxis)
plt.show()


fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = plot_acf(df['Food'], lags=20, ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(df['Food'], lags=20, ax=ax2)
fig.show()

fig = plt.figure(figsize=(12,8))

ax1 = fig.add_subplot(211)
fig = plot_acf(df['Fuel'], lags=20, ax=ax1)
ax2 = fig.add_subplot(212)
fig = plot_pacf(df['Fuel'], lags=20, ax=ax2)
fig.show()

# #Adfuller
result = adfuller(df['Food'])
print(f'Adfuller Result FOOD: The series is {"not " if result[1] >= 0.05 else ""}stationary')
result = adfuller(df['Fuel'])
print("Adfuller /       pValue ->  ",result[1],f' /      Result FUEL: The series is {"not " if result[1] >= 0.05 else ""}stationary')

# #KPSS
statistic, p_value, n_lags, critical_values = kpss(df['Food'])
print(f'KPSS Result FOOD: The series is {"not " if p_value < 0.05 else ""}stationary')
statistic, p_value, n_lags, critical_values = kpss(df['Fuel'])
print("KPSS    /       pValue ->  ",p_value,f' /  Result FUEL: The series is {"not " if p_value < 0.05 else ""}stationary')


#Shapiro
print("Shapiro Food:  statistic -> ", stats.shapiro(df['Food'])[0], " / pValue -> ", stats.shapiro(df['Food'])[1], f' \n                                 / Result (alpha=0.05) : The series is {"not " if stats.shapiro(df["Food"])[1] <= 0.05 else ""}normally distributed')
print("Shapiro Fuel:  statistic -> ", stats.shapiro(df['Fuel'])[0], " / pValue -> ", stats.shapiro(df['Fuel'])[1], f' \n                                / Result (alpha=0.05) : The series is {"not " if stats.shapiro(df["Fuel"])[1] <= 0.05 else ""}normally distributed')

#Boxpierce
#https://support.minitab.com/fr-fr/minitab/18/help-and-how-to/modeling-statistics/time-series/how-to/arima/interpret-the-results/all-statistics-and-graphs/modified-box-pierce-ljung-box-chi-square-statistics/
print("BoxPierce Food:  / pValue -> ", acorr_ljungbox(df['Food'])[1][0], f'\n                                 / Result (alpha=0.05) : The data are {"not " if acorr_ljungbox(df["Food"])[1][0] <= 0.05 else ""}independently distributed')
print("BoxPierce Fuel:  / pValue -> ", acorr_ljungbox(df['Fuel'])[1][0], f'\n                                 / Result (alpha=0.05) : The data are {"not " if acorr_ljungbox(df["Fuel"])[1][0] <= 0.05 else ""}independently distributed')