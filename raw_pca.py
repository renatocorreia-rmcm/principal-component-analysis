import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity

"""
    LOAD PRE-PROCESSED DATA
"""

df = pd.read_csv("data/breast_cancer_wisconsis/breast_cancer_wisconsis_processed_test.csv")

targets = df["diagnosis"].to_numpy()
features = df.drop(columns=["diagnosis"])

features_benign = features[targets == 0]
features_malignum = features[targets == 1]

"""
    FIND PRINCIPAL COMPONENTS   
"""

pca_benign = PCA()
pca_malignum = PCA()

projected_benign = pca_benign.fit_transform(features_benign)
projected_malignum = pca_malignum.fit_transform(features_malignum)

"""
    PRINCIPAL COMPONENTS EXPLAINED VARIANCE
"""

cumulative_benign = np.cumsum(pca_benign.explained_variance_ratio_)
cumulative_malignum = np.cumsum(pca_malignum.explained_variance_ratio_)

fig, axes = plt.subplots()

axes.scatter(  # benign
    np.arange(1, len(cumulative_benign) + 1),
    cumulative_benign,
    label="benign",
    c='c',
    alpha=0.7
)
axes.scatter(  # malignum
    np.arange(1, len(cumulative_malignum) + 1),
    cumulative_malignum,
    label="malignum",
    c='r',
    alpha=0.7
)

axes.set_xlabel("Number of Principal Components")
axes.set_ylabel("Cumulative Explained Variance")

plt.xticks(np.arange(1, len(features.columns) + 1))
axes.grid()
axes.legend()
plt.show()
plt.close(fig)

"""
    DATA PROJECTIONS
"""

# todo:
"""
    print PCs of Benign and Malign to verify if 
    they indeed point to different directions
    
    can check cosine distance or scalar product in each dimension (measuring similarity)
    
    <u,v> = |u|*|v|*cos(u,v)
    cos(u, v) = (u*v)/(|u|*|v|)

"""

PCs_cosine_similarity = []

for i in range(features.shape[1]):
    v1 = pca_benign.components_[i]
    v2 = pca_malignum.components_[i]

    sim = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    PCs_cosine_similarity.append(np.abs(sim))

fig, ax = plt.subplots()

ax.scatter(x=np.arange(len(PCs_cosine_similarity)), y=PCs_cosine_similarity)
plt.yticks(PCs_cosine_similarity)
ax.grid()
plt.show()

# todo:
"""
    plotar dados em diferentes bases para mostrar erro de representação em cada base
    
    mudança de base canonica pra PCs de benign: B (Retangular)
    mudança de PCs de benign pra base canonica: B^-1 (não existe)
    
    mudança de base canonica pra PCs de malignum: M (Retangular) 
    mudança de PCs de malignum pra base canonica: M^-1 (não existe) 
"""

# 1D

fig, axes = plt.subplots(2)

benign_benignBasis = projected_benign[:, 0]
benign_malignumBasis = None  # compute this

malignum_malignumBasis = projected_malignum[:, 0]
malignum_benignBasis = None  # compute this

# benign change of basis
axes[0].hist(benign_benignBasis, alpha=0.5, label="Benign", color='c')
axes[0].hist(benign_malignumBasis, alpha=0.5, label="Benign", color='c')

axes[1].hist(x_Malignum, alpha=0.5, label="Malignant", color='r')

for ax in axes:
    ax.set_xlabel("PC1")
    ax.set_ylabel("Frequency")
    ax.legend()

plt.show()

# 2D

fig, axes = plt.subplots(2)

x_benign, y_benign = projected_benign[:, :2].T
axes[0].scatter(x_benign, y_benign, c='c', alpha=0.4)

x_malignum, y_malignum = projected_malignum[:, :2].T
axes[1].scatter(x_malignum, y_malignum, c='r', alpha=0.4)

for ax in axes:
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")

plt.show()

# 3D
