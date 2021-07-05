import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
import seaborn as sns; sns.set()
import scipy.stats as sc;


# iris = pd.read_csv("iris.csv", sep=",")
# print('#####PARTIE A#####')
# print('')
# print('------Question 1------')
# print(iris.shape)
#
#
# print('')
# print('------Question 2------')
# print('Voir Histogrammes')
# sns.displot(iris['sepal_length'])
# plt.title("Length histogram of the sepals")
# plt.show()
#
# sns.displot(iris['sepal_width'])
# plt.title("Width histogram of the sepals")
# plt.show()
#
# sns.displot(iris['petal_length'])
# plt.title("Length histogram of the petals")
# plt.show()
#
# sns.displot(iris['petal_width'])
# plt.title("width histogram of the petals")
# plt.show()
#
# print('')
# print('------Question 3------')
# print(iris.corr())
#
# print('')
# print('------Question 4------')
# print('voir Histos')
# sns.pairplot(iris)
# plt.show()
#
# sns.heatmap(iris.corr())
# plt.show()
#
# print('')
# print('------Question 5------')
#
#
def r_confiance(r,n):
    Z = (np.log(1+r) - np.log(1-r))/2
    S = math.sqrt(1/(n-3))

    Zi = Z - 1.96*S
    Zs = Z + 1.96*S

    ICi = (math.exp(2*Zi)-1)/(math.exp(2*Zi)+1)
    ICs = (math.exp(2*Zs)-1)/(math.exp(2*Zs)+1)

    return [ICi,ICs]
#
# print('sepal_width / sepal_length : ')
# print(r_confiance(-0.117570,n=150))
# print('')
#
# print('petal_length / sepal_length : ')
# print(r_confiance(0.871754,n=150))
#
# print('')
# print('petal_width / sepal_length : ')
# print(r_confiance(0.817941,n=150))
#
# print('')
# print('petal_length / sepal_width : ')
# print(r_confiance(-0.428440,n=150))
#
# print('')
# print('petal_width / sepal_width : ')
# print(r_confiance(-0.366126,n=150))

#
# print('')
# print('')
# print('#####PARTIE B#####')
#
# mansize = pd.read_csv("mansize.csv", sep=";")
#
# print('')
# print('------Question 1------')
# print(mansize.shape)
#
# print('')
# print('------Question 2------')
# print(mansize.describe())
#
# print('')
# print('------Question 3------')
#
# corr = mansize.corr()
#
# sns.pairplot(mansize)
# plt.show()
#
# sns.heatmap(corr)
# plt.show()
#
# print(corr)
#
# def r_confiance_inf(r,n):
#     Z = (np.log(1+r) - np.log(1-r))/2
#     S = math.sqrt(1/(n-3))
#
#     Zi = Z - 1.96*S
#
#     ICi = (math.exp(2*Zi)-1)/(math.exp(2*Zi)+1)
#
#     return ICi
#
# def r_confiance_sup(r,n):
#     Z = (np.log(1+r) - np.log(1-r))/2
#     S = math.sqrt(1/(n-3))
#
#     Zs = Z + 1.96*S
#
#     ICs = (math.exp(2*Zs)-1)/(math.exp(2*Zs)+1)
#
#     return ICs



print('------INF------')
corr_copy_inf = corr.copy()

for i in range(len(corr)):
    for j in range(len(corr)):
        if i!=j:
            corr_copy_inf.iloc[i,j] = r_confiance_inf(corr_copy_inf.iloc[i,j],161)

print(corr_copy_inf)

print('------SUP------')
corr_copy_sup = corr.copy()

for i in range(len(corr)):
    for j in range(len(corr)):
        if i!=j:
            corr_copy_sup.iloc[i,j] = r_confiance_sup(corr_copy_sup.iloc[i,j],161)

print(corr_copy_sup)

print('')
print('')
print('#####PARTIE C#####')

weather = pd.read_csv('weather.csv',sep=';')
# print(weather.describe())
#
# print('')
#
# sns.displot(weather['Outlook'])
# plt.title('Outlook observations in 193 cities')
# plt.show()
#
# sns.displot(weather['Humidity'])
# plt.title('Humidity observations in 193 cities')
# plt.show()
#
# sns.displot(weather['Temperature'])
# plt.title('Temperature observations in 193 cities')
# plt.show()

print("")
print("-----")
crosstab = pd.crosstab(weather["Temperature"],weather["Humidity"])
print(crosstab)

print("")
print("-----")

print(sc.chi2_contingency(crosstab))
