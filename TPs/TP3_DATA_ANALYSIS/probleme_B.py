import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np



df = pd.read_csv("iris.csv",sep=",",index_col=['Class'])
scaler = StandardScaler()

Colors =['b' for i in range(50)] + ['g' for i in range(50)] + ['r' for i in range(50)]

df_normalized = scaler.fit_transform(df)
df_normalized = pd.DataFrame(df_normalized)

pca = PCA(n_components=4)
pca.fit(df_normalized)
pca_data = pca.transform(df_normalized)

per_var = np.round(pca.explained_variance_ratio_*100,decimals=1)
labels = ['PC' + str(x) for x in range(1,len(per_var)+1)]
plt.bar(x=range(1,len(per_var)+1),height=per_var)
plt.ylabel('Percentage of Explained Variance')
plt.xlabel('Principal Component')
plt.title('Scree Plot')
plt.show()

pca_df = pd.DataFrame(pca_data, columns=labels)

plt.scatter(pca_df.PC1,pca_df.PC2, color = Colors)
plt.title("PCA graph")
plt.xlabel('PC1 - {0}%'.format(per_var[0]))
plt.ylabel('PC2 - {0}%'.format(per_var[1]))
plt.grid(linestyle='--')
plt.show()
