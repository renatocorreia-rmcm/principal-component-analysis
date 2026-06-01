import numpy as np
from matplotlib import pyplot as plt

from PCAClassifier import Classe


class PCAVisualizer:

    def __init__(self, classes: list[Classe], components_amount_to_show = None, dimensions_amount_to_show = None):
        """
        :param classes: List of Classe object
        :param components_amount_to_show:  First N Principal Components plotted
        :param dimensions_amount_to_show: Fraction of Principal Components dimensions to show
        """

        self.classes = classes

        self.components_amount: int = classes[0].components.shape[0]
        self.dimensions_amount: int = classes[0].components.shape[1]

        # If amount is not specified, show all
        if components_amount_to_show is None:
            components_amount_to_show = self.components_amount
        if dimensions_amount_to_show is None:
            dimensions_amount_to_show = self.dimensions_amount

        self.components_amount_to_show = min(components_amount_to_show, self.components_amount)  # range checked
        self.dimensions_amount_to_show = min(dimensions_amount_to_show, self.dimensions_amount)

    def cumulative_variance(self):
        """
        Plot cumulative explained variance X amount of PCs used
        for each class
        """

        x = range(1, 1+self.components_amount)

        for i, c in enumerate(self.classes):  # for each class
            cumulative_variance = np.cumsum(c.explained_variances)
            plt.plot(x, cumulative_variance, label=f"{i}")

        # Labels
        plt.title("Explained variance")
        plt.ylabel("Cumulated Variance")
        plt.xlabel("Number of Components Used")

        plt.grid()
        plt.legend(title='Class')

        plt.show()

    def components(self):
        """
        Plot N graphs, each one for the i-th component
        """

        # Load all classes principal components (3D tensor)
        # Each Matrix contains a class; Each Row contains a PC; Each Column contains a coordinate
        principal_components_matrix = []
        for c in self.classes:
            # using abs to highlight each coordinate contribution, directly or inversely correlated
            abs_components = np.abs(c.components)
            principal_components_matrix.append(abs_components)

        principal_components_matrix = np.array(principal_components_matrix)

        # Generate the sampled coordinates ahead of time
        coordinates_samples = np.linspace(
            start=0,
            stop=self.dimensions_amount - 1,
            num=self.dimensions_amount_to_show
        )
        coordinates_samples = np.round(coordinates_samples).astype(int)

        # FIX: Calculate global bounds ONLY from the subset of data being displayed
        display_sub_matrix = principal_components_matrix[:, :self.components_amount_to_show, coordinates_samples]
        min_principal_component = float(np.min(display_sub_matrix))
        max_principal_component = float(np.max(display_sub_matrix))

        # One plot for each component. Each class gets in a row
        for component in range(self.components_amount_to_show):

            grid_data = principal_components_matrix[:, component, coordinates_samples]  # All classes, i-th Component, N coordinates to sample

            plt.imshow(
                grid_data,
                vmin=min_principal_component,
                vmax=max_principal_component,
                aspect='auto'  # cells are flexible and stretch to fill the space
            )

            plt.gca().set_box_aspect(1)  # Force the total graph box to be a perfect square

            plt.colorbar()

            xticks = range(self.dimensions_amount_to_show)
            plt.xticks(ticks=xticks, labels=coordinates_samples)
            plt.yticks(range(len(self.classes)))

            plt.title(f"Principal component {component + 1}")
            plt.ylabel(f"Class")
            plt.xlabel(f"Coordinate")
            plt.show()

    def components_similarity(self):
        """
        Plot N graphs, each one for the i-th class
        Each row is a class, each column is a Component Cosine Distance

        """

        for i, c in enumerate(self.classes):

            plt.imshow(
                c.similarity_matrix[:,:self.components_amount_to_show],
                vmin=0,
                vmax=1,
                aspect='auto'  # cells are flexible and stretch to fill the space

            )
            plt.gca().set_box_aspect(1)  # Force the total graph box to be a perfect square

            plt.colorbar()

            plt.title(f"Abs Cosine Similarity to CLASS {i}")
            plt.ylabel(f"Class")
            plt.xlabel(f"Component")

            xticks = np.array(range(self.components_amount_to_show))
            plt.xticks(ticks=xticks, labels=xticks+1)
            plt.yticks(range(len(self.classes)))

            plt.show()
