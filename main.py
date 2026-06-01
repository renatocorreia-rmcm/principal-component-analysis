import pandas as pd

from PCAClassifier import PCAClassifier
from PCAVisualizer import PCAVisualizer

# todo: Implement PCA
# todo: Implement C-PCA or use lib at https://github.com/abidlabs/contrastive

# todo: Implement Representation Error (v - v')
#   Implement basis-change (f(v) = v')

# todo: Increment PCA visualizer to analyse result

# todo: Documentation


"""
    LOADING DATA
"""
print()  # skip line for clean log

print("Initializing Data...")
fashion_mnist = pd.read_csv("data/fashion_mnist/train.csv")
breast_cancer_wisconsin = pd.read_csv("data/breast_cancer_wisconsin/train.csv")

dataset_to_use = breast_cancer_wisconsin


"""
    PCA
"""
print()  # skip line for clean log

print("Initializing Classifier...")
classifier = PCAClassifier(df=dataset_to_use)

print("Computing principal components...")
classifier.compute_principal_components()

print("Computing principal components similaritys...")
classifier.compute_principal_components_similarity()


"""
    PCA RESULT ANALYSIS
"""
print()  # skip line for clean log

print("Initializing Visualizer...")
visualizer = PCAVisualizer(
    classes=classifier.classes,
    components_amount_to_show=5,
    dimensions_amount_to_show=10
)

print("Plotting Cumulative Explained Variances...")
visualizer.cumulative_variance()

print("Plotting Principal Components...")
visualizer.components()

print("Plotting Components Cosine Similarity Matrix..")
visualizer.components_similarity()

