import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import numpy as np
from sklearn.manifold import MDS
from sklearn.manifold import LocallyLinearEmbedding
from sklearn import manifold

alon = pd.read_csv('alon.csv', sep= ';')




print(alon.shape)

print(alon.describe())
