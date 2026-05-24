import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("breast_cancer_wisconsis.csv")

labels = df["diagnosis"]
df = df.drop(columns=["id", "diagnosis"])


scaler = StandardScaler()  # z score scalling

df_scaled = pd.DataFrame(
    scaler.fit_transform(df),
    columns=df.columns
)


n_components = 3
pca = PCA(n_components=n_components)  # use n_componets =
projected_points = pca.fit_transform(df.to_numpy())

print("variance explained by each principal componet found:")
print(pca.explained_variance_)

print("plotting variance decay along amount of principal components used to represent the data")
fig, ax = plt.subplots()
components_variances = pca.explained_variance_
ax.plot(np.arange(len(components_variances)), components_variances)
plt.xticks(np.arange(len(components_variances)))
plt.show()

labels = labels.to_numpy()
labels[labels == "B"] = 'c'
labels[labels == "M"] = 'r'

fig = plt.figure()

if n_components == 2:
    print("plotting 2d points")

    ax = fig.add_subplot()
    ax.set_aspect("equal")

    x, y = projected_points.T
    ax.scatter(x, y, color=labels)
    plt.show()


if n_components == 3:
    print("plotting 3d points")
    ax = fig.add_subplot(projection='3d')
    ax.set_aspect("equal")

    x, y, z = projected_points.T
    ax.scatter(x, y, z, color=labels)
    plt.show()
