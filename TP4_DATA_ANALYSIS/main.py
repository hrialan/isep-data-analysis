import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import scipy as sc
import seaborn as sns

from sklearn import datasets
from sklearn.cluster import KMeans, SpectralClustering, MeanShift, FeatureAgglomeration, DBSCAN, AffinityPropagation
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import confusion_matrix, silhouette_score

#Problem A

#On importe les données
irisData = datasets.load_iris()
df = pd.DataFrame(irisData['data'], columns=irisData['feature_names'])

label = []
for elt in irisData['target']:
    label.append(irisData['target_names'][elt])

label_copy = label.copy()

#On normalise et applique le PCA
pca = PCA(n_components=2)
PCA_val = pca.fit_transform(df)
df_iris_PCA = pd.DataFrame(PCA_val, columns=['PC1', 'PC2'])

#On visualise les données projetées à l'aide de couleurs
colors = []
for elt in irisData['target']:
    if elt == 0:
        colors.append('navy')
    elif elt == 1:
        colors.append('turquoise')
    else:
        colors.append('darkorange')

plt.scatter(df_iris_PCA['PC1'], df_iris_PCA['PC2'], color=colors)

plt.title('PCA Graph with true class label')
plt.xlabel('PC1')
plt.ylabel('PC2')

# legend
import matplotlib.patches as mpatches
label_1 = mpatches.Patch(color='navy', label='Setosa')
label_2 = mpatches.Patch(color='turquoise', label='Versicolor')
label_3 = mpatches.Patch(color='darkorange', label='Virginica')
plt.legend(handles=[label_1,label_2,label_3])

plt.grid()
plt.show()

#Kmeas algorithm (Or others this is the same code if we change the algorithm name (ex: Gaussian Mixture))
kmeans = KMeans(n_clusters=3, n_init=5, max_iter=300, random_state=2).fit(df)
kmeans.score(df)
prediction = kmeans.predict(df)


#Mise en couleur en fonction des predictions
colors = []
for elt in prediction:
    if elt == 0:
        colors.append('navy')
    elif elt == 1:
        colors.append('turquoise')
    else:
        colors.append('darkorange')


#visualisation
fig = plt.figure(figsize=(15, 10))
plt.scatter(df_iris_PCA['PC1'], df_iris_PCA['PC2'], color=colors)

plt.title('PCA Graph with KMean clustering')
plt.xlabel('PC1')
plt.ylabel('PC2')


for x_pos, y_pos, label_copy in zip(df_iris_PCA['PC1'], df_iris_PCA['PC2'], label):
    plt.annotate(label_copy,             # The label for this point
                xy=(x_pos, y_pos), # Position of the corresponding point
                xytext=(- 7, 0),     # Offset text by 7 points to the right
                textcoords='offset points', # tell it to use offset points
                ha='right',         # Horizontally aligned to the left
                va='center')       # Vertical alignment is centered


label_1 = mpatches.Patch(color='navy', label='Cluster 1')
label_2 = mpatches.Patch(color='turquoise', label='Cluster 2')
label_3 = mpatches.Patch(color='darkorange', label='Cluster 3')
plt.legend(handles=[label_1,label_2,label_3])

plt.grid()
plt.show()


#confusion matrice
x_true = []
for elt in irisData['target']:
    x_true.append(irisData['target_names'][elt])

x_pred = []
for elt in prediction:
    if elt == 1:
        x_pred.append('setosa')
    elif elt == 0:
        x_pred.append('versicolor')
    else:
        x_pred.append('virginica')

#Silhouette score
#print(silhouette_score(df,prediction))


#Problem C

#on importe les données
wdbc_data = pd.read_csv('wdbc/wdbc.data',names=["col_" + str(i) for i in range(32)])
wdbc_diagnosis = wdbc_data['col_1'];
wdbc_data = wdbc_data.drop(['col_0','col_1'],axis=1)


#On normalise et applique le PCA
pca = PCA(n_components=2)
PCA_val = pca.fit_transform(wdbc_data)
wdbc_PCA = pd.DataFrame(PCA_val, columns=['PC1', 'PC2'])

#Visualisation
Colors = ['navy','turquoise', 'darkorange','green ','red','blue']

true_colors = []
for elt in wdbc_diagnosis:
    if elt == 'B':
        true_colors.append(Colors[0])
    else:
        true_colors.append(Colors[1])

plt.scatter(wdbc_PCA['PC1'], wdbc_PCA['PC2'], c=true_colors)
plt.title('PCA Graph WDBC colored by diagnosis')
pc1 = int(round(pca.explained_variance_ratio_[0] * 100))
pc2 = int(round(pca.explained_variance_ratio_[1] * 100))
plt.xlabel('PC1 - {}%'.format(pc1))
plt.ylabel('PC2 - {}%'.format(pc2))
label_1 = mpatches.Patch(color='navy', label='malignant')
label_2 = mpatches.Patch(color='turquoise', label='benign')
plt.legend(handles=[label_1,label_2])
plt.grid()
plt.show()

#On applique l'algorithme de clustering
kmeans = KMeans(n_clusters=2).fit(wdbc_data)
#kmeans = MeanShift(n_clusters=2, n_init=5, random_sate=2).fit(wdbc_data)

prediction = kmeans.fit_predict(wdbc_data)


prediction_color = []
for elt in prediction:
   prediction_color.append(Colors[elt])


plt.scatter(wdbc_PCA['PC1'], wdbc_PCA['PC2'], c=prediction_color)
plt.title('PCA Graph WDBC with KMEAN clustering')
pc1 = int(round(pca.explained_variance_ratio_[0] * 100))
pc2 = int(round(pca.explained_variance_ratio_[1] * 100))
plt.xlabel('PC1 - {}%'.format(pc1))
plt.ylabel('PC2 - {}%'.format(pc2))

plt.grid()
plt.show()

print("-------------")
prediction_diagnosis = []
for elt in prediction:
    if elt == 1:
        prediction_diagnosis.append('M')
    else:
        prediction_diagnosis.append('B')

print(confusion_matrix(wdbc_diagnosis,prediction_diagnosis))
print("-------------")
print(silhouette_score(wdbc_data,prediction))
