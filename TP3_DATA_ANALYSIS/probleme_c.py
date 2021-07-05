import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import manifold

golub = pd.read_csv('golub_data.csv',sep=',')

scaler = StandardScaler()
golub = scaler.fit_transform(golub)
golub = pd.DataFrame(golub)

golub_class = pd.read_csv('golub_class2.csv',sep=',')

pca = PCA(n_components=72)
pca.fit(golub)
pca_data = pca.transform(golub)

per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)
labels = ['PC' + str(x) for x in range(1,len(per_var)+1)]
plt.bar(x=range(1,len(per_var)+1),height=per_var)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

pca_df = pd.DataFrame(pca_data, columns=labels)



plt.scatter(pca_df.PC1,pca_df.PC2)
plt.title("PCA graph")
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
plt.grid(linestyle='--')
plt.show()




