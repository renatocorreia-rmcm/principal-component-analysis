import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# todo: bifurcates approaches:
#       pca to simplify classification by a model
#           svd, linear_regression
#       pca to properly classificate
#           PCA each class separetely. then classify sample by represetation error
#           this is interesting because currently PC1 of whole dataset doesnt differentiate between classes at all

# todo: try on a image dataset

"""
    LOAD PRE-PROCESSED DATA
"""

df = pd.read_csv("data/breast_cancer_wisconsis/breast_cancer_wisconsis_processed_test.csv")

targets = df["diagnosis"].to_numpy()
features = df.drop(columns=["diagnosis"])
colors = np.where(targets == 0, "c", "r")


"""
    FIND PRINCIPAL COMPONENTS   
"""

pca = PCA()
projected_points = pca.fit_transform(features)

# cumulative variance
cumulative = np.cumsum(pca.explained_variance_ratio_)

fig, ax = plt.subplots()
ax.scatter(np.arange(1, len(cumulative)+1), cumulative)
ax.set_xlabel("Number of Principal Components")
ax.set_ylabel("Cumulative Explained Variance")
plt.xticks(np.arange(len(cumulative)))
plt.yticks(cumulative)
ax.grid()
plt.show()


"""
    PLOTTING
"""


# colors

# 1D plot

fig, ax = plt.subplots()

x_Benign = projected_points[targets == 0, 0]
x_Malignant = projected_points[targets == 1, 0]

ax.hist(x_Benign, alpha=0.5, label="Benign", color='c')
ax.hist(x_Malignant, alpha=0.5, label="Malignant", color='r')

ax.set_xlabel("PC1")
ax.set_ylabel("Frequency")
ax.legend()

plt.show()


# 2D plot
fig, ax = plt.subplots()

x, y = projected_points[:, :2].T
ax.scatter(x, y, color=colors, alpha=0.4)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")

plt.show()

# 3D plot
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

x, y, z = projected_points[:, :3].T
ax.scatter(x, y, z, color=colors, alpha=0.5)

ax.set_xlabel("PC1")
ax.set_ylabel("PC2")
ax.set_zlabel("PC3")

plt.show()