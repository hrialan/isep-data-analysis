import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import numpy as np

#Partie A Titanic
#on ouvre les données "titanic_train.csv" et "titanic_test.csv"

titanic_train = pd.read_csv('titanic_train.csv')
titanic_test = pd.read_csv('titanic_test.csv')

print("titanic_train shape :", titanic_train.shape)
print("titanic_test shape :",titanic_test.shape)

print("")
attributes = list(titanic_test)
print("Attributes of the data set :",attributes)

print("")
print("DType")
for elt in attributes:
    print(elt," ", np.dtype(titanic_test[elt]))

#Missing Datas
print("")
print("Missing datas:")
print("")
for elt in attributes:
    print(elt," : ", sum(pd.isnull(titanic_test[elt])))


print("")
titanic_train_drop_na_age = titanic_train.dropna(axis = 0)
plt.hist(titanic_train_drop_na_age['Age'], rwidth = 0.7)
plt.grid()
plt.xlabel('Age')
plt.ylabel('Number of passenger')
plt.title('Passenger age histogram')
plt.show()

print("")
print("dead passengers : ", round((1 - titanic_train['Survived'].mean()),3)*100, " %")


print("")
print("Passenger per classes :")
print(titanic_train['Pclass'].value_counts())

label_value = zip(['3','1','2'],list(titanic_train['Pclass'].value_counts()))
labels=[]
for Pclass, Pnumber in label_value:
    value= 'Class ' + Pclass + ' - ' + str(Pnumber) + ' Passengers'
    labels.append(value)


print(labels)

plt.pie(titanic_train['Pclass'].value_counts().values, labels=labels)
plt.title('Piechart of the passengers classes')
plt.show()
