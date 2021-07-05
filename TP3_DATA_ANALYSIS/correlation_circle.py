import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.cm as cm
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X = pd.read_csv("iris.csv",sep=",",index_col=['Class'])
scaler.fit(X)
X=scaler.transform(X)

mypca = PCA(n_components=4)

mypca.fit(X)

data_sortie= mypca.fit_transform(X)

Colors =['b' for i in range(50)] + ['g' for i in range(50)] + ['r' for i in range(50)]

def biplot(x=None,y=None,score=None,coeff=None,coeff_labels=None,circle='T',cat=None,cmap="viridis"):
    if score is not None : x = score
    if x.shape[1]>1 :
        xs = x[:,0]
        ys = x[:,1]
    else :
        xs = x
        ys = y
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    #x_c = xs * scalex
    #y_c = ys * scaley
    temp = (xs - xs.min())
    x_c = temp / temp.max() * 2 - 1
    temp = (ys - ys.min())
    y_c = temp / temp.max() * 2 - 1
    if cat is None : cat = [0]*len(xs)
    elif len(pd.Series(cat)) == 1 : cat = list(pd.Series(cat))*len(xs)
    elif len(pd.Series(cat)) != len(xs) : print("Warning ! Nombre anormal de catégories !")
    cat = pd.Series(cat).astype("category")
    fig = plt.figure(figsize=(6,6),facecolor='w')
    ax = fig.add_subplot(111)
    # Affichage des points
    ax.scatter(x_c,y_c, c = cat.cat.codes,cmap=cmap)
    if coeff is not None :
        if (circle == 'T') :
            x_circle = np.linspace(-1, 1, 100)
            y_circle = np.linspace(-1, 1, 100)
            X, Y = np.meshgrid(x_circle,y_circle)
            F = X**2 + Y**2 - 1.0
            #fig, ax = plt.subplots()
            plt.contour(X,Y,F,[0])
        n = coeff.shape[0]
        for i in range(n):
            plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5,
                      head_width=0.05, head_length=0.05)
            if coeff_labels is None:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'g', ha = 'center', va = 'center')
            else:
                plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, coeff_labels[i], color = 'g', ha = 'center', va = 'center')
    plt.xlim(-1.2,1.2)
    plt.ylim(-1.2,1.2)
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.grid(linestyle='--')

biplot(score=data_sortie[:,0:2],coeff=np.transpose(mypca.components_[0:2, :]),coeff_labels=['SepalLength','SepalWidth','PetalLength','PetalWidth'],cat=Colors)
plt.show()
