import numpy as np
import pandas as pd
import numpy.typing as npt
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity


class Classe:
    """
    Subset of dataset with the same label

    """
    def __init__(self, df: pd.DataFrame):
        self.df = df.drop(columns=['label'], errors='ignore')

        self.pca = None
        self.components = None

        self.explained_variances = None  # array for [component i]

        self.similarity_matrix = None  # similarity between principal component j of self.class and class_i

    def compute_principal_components(self):
        self.pca = PCA()
        self.pca.fit(self.df)
        self.components = self.pca.components_
        self.explained_variances = self.pca.explained_variance_ratio_

    def compute_principal_components_similarity(self, all_classes_components: npt.NDArray):
        """

        :param all_classes_components: list of matrixes for [component i x feature k] of class k
        :return matrix for [class i similarity x component j]
        """

        num_classes = all_classes_components.shape[0]
        num_components = self.components.shape[0]

        # Shape: (num_classes, num_components)
        self.similarity_matrix = np.zeros((num_classes, num_components))

        for k in range(num_classes):  # For each class
            for i in range(num_components):  # For each component
                # Reshape to 2D (1, n_features) as required by cosine_similarity
                vec_self = self.components[i].reshape(1, -1)
                vec_other = all_classes_components[k, i].reshape(1, -1)

                self.similarity_matrix[k, i] = np.abs(cosine_similarity(vec_self, vec_other)[0, 0])


class PCAClassifier:

    def __init__(self, df: pd.DataFrame):
        self.df = df

        # array of dataframes, one for each class
        self.classes = [Classe(group) for _, group in df.groupby('label')]

        self.explained_variances = []  # matrix class_i x component j

    def compute_principal_components(self):
        for c in self.classes:
            c.compute_principal_components()

    def compute_contrastive_principal_components(self):
        pass

    def compute_principal_components_similarity(self):
        all_components = np.array([c.components for c in self.classes])

        for c in self.classes:
            c.compute_principal_components_similarity(all_components)

