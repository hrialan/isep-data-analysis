import matplotlib as plt
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from math import sqrt, pi, exp

df = pd.read_csv("malnutrition.csv")


print(np.mean(df["BOW"]))

print("---------------")

print(np.var(df["BOW"]))

print("---------------")

print(sqrt(np.var(df["BOW"])))
print(np.std(df["BOW"]))

print("---------------")

print(np.median(df["BRW"]))

print("---------------")
print(df['AUD'])

plt.hist(df["AUD"])
plt.title("Histogramme des volumes du noyau auditif chez des chauves-souris")
plt.xlabel("Volume du noyau auditif en mm3")
plt.ylabel("Fréquence")
plt.grid()
plt.show()




df = pd.DataFrame({'Notes': [6 for i in range(10)]
                           + [8 for i in range(12)]
                           + [9 for i in range(48)]
                           + [10 for i in range(23)]
                           + [11 for i in range(24)]
                           + [12 for i in range(48)]
                           + [13 for i in range(9)]
                           + [14 for i in range(14)]
                           +[17 for i in range(22)]
                   })

print(df['Notes'].mean())
print(df['Notes'].median())
print(df['Notes'].mode())

plt.hist(df["Notes"],rwidth = 0.8)
plt.grid()
plt.show()
