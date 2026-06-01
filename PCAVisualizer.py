import numpy as np
from matplotlib import pyplot as plt


class PCAVisualizer:

    def __init__(self, classes: list):
        self.classes = classes

    def cumulative_variance(self):
        print("plotting principal components explained variance...\n")
        for i, c in enumerate(self.classes):
            cumulative_variance = np.cumsum(c.explained_variances)
            plt.plot(np.arange(cumulative_variance.shape[0]) + 1, cumulative_variance, label=f"{i}")
        plt.grid()
        plt.legend()
        plt.show()

    def components_similarity(self):
        print("Ploting components cosine similarity")
        for i, c in enumerate(self.classes):
            plt.imshow(c.similarity_matrix[:,:20])
            plt.show()
