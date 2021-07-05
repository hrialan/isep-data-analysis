import pydotplus
import collections
from IPython.display import Image
from sklearn.tree import export_graphviz
from sklearn import tree
import pandas as pd


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

#Ajout colonne Has cabin
titanic_test['cabin2'] = titanic_test["Cabin"].apply(lambda x: 0 if x==0 else 1)
titanic_train['cabin2'] = titanic_train["Cabin"].apply(lambda x: 0 if x==0 else 1)

#Ajout colonne Has cabin
titanic_test['cabin2'] = titanic_test["Cabin"].apply(lambda x: 0 if x==0 else 1)
titanic_train['cabin2'] = titanic_train["Cabin"].apply(lambda x: 0 if x==0 else 1)

#Ajout colonne
titanic_test['closeFamily'] = titanic_test["SibSp"] + titanic_test["Parch"]
titanic_train['closeFamily'] = titanic_train["SibSp"] + titanic_train["Parch"]

clf=tree.DecisionTreeClassifier(max_depth=3)
data_feature_name = ['Child','Sex_cat','Fare2','cabin2','closeFamily', 'Pclass']

clf=clf.fit(titanic_train[data_feature_name], titanic_train['Survived'])

dot_data = tree.export_graphviz(clf, out_file=None, feature_names=data_feature_name,class_names=['Dead', 'Survived'],filled=True,rounded=True)

graph = pydotplus.graph_from_dot_data(dot_data)

Image(graph.create_png())#toplotatree
Image(graph.write_png('./tree.png'))#tosavetheplot

acc_decision_tree = round(clf.score(titanic_train[data_feature_name], titanic_train['Survived']) * 100, 2)
print("Accuracy of the decision tree: ",acc_decision_tree)
