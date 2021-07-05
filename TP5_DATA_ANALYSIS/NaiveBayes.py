import pandas as pd
import matplotlib.pyplot as plt
import scipy as sc
import numpy as np


#Partie B  Naive Bayes
titanic_train = pd.read_csv('titanic_train.csv')
titanic_test = pd.read_csv('titanic_test.csv')

#Ajout 0 sur passagers sans canines
titanic_train['Cabin'] = titanic_train['Cabin'].fillna(0)
titanic_test['Cabin'] = titanic_test['Cabin'].fillna(0)

#Suppression passagers sans ages
titanic_train = titanic_train.dropna(axis=0)
titanic_test = titanic_test.dropna(axis=0)


#Ajout colonne Child
titanic_train['Child'] = titanic_train["Age"].apply(lambda x: 1 if x<18 else 0)
titanic_test['Child'] = titanic_test["Age"].apply(lambda x: 1 if x<18 else 0)


#Ajout colonne Sex_cat
titanic_train['Sex_cat'] = titanic_train["Sex"].apply(lambda sex: 1 if str(sex)=='female' else 0)
titanic_test['Sex_cat'] = titanic_test["Sex"].apply(lambda sex: 1 if str(sex)=='female' else 0)

#Ajout colonne Fare2
def fareCategory(fare):
    if (fare < 10):
        return 1
    elif (fare < 20):
        return 2
    elif (fare < 30):
        return 3
    else:
        return 4

titanic_train['Fare2'] = titanic_train["Fare"].apply(lambda fare: fareCategory(fare))
titanic_test['Fare2'] = titanic_test["Fare"].apply(lambda fare: fareCategory(fare))



# needed imports
from sklearn import metrics
from sklearn.naive_bayes import GaussianNB


#classifier choice
gnbModel=GaussianNB()

#choice of the training set , considered attributes and variable to predict
gnbModel.fit(titanic_train[['Child','Sex_cat']], titanic_train['Survived'])

#expected results are stored in a separate vector
expected =titanic_train['Survived']

#predictions on the training set
predicted = gnbModel.predict(titanic_train[['Child','Sex_cat']]) #displaying relevant metrics
print(metrics.classification_report(expected, predicted))



#same when applying the model to the test set
expected =titanic_test['Survived']
predicted = gnbModel.predict(titanic_test[['Child','Sex_cat']])
print(metrics.classification_report(expected, predicted))

