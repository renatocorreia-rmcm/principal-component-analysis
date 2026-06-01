import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PCAClassifier import PCAClassifier
from PCAVisualizer import PCAVisualizer

print("loading dataset...\n")
fashion_mnist = pd.read_csv("data/fashion_mnist/train.csv")
breast_cancer_wisconsin = pd.read_csv("data/breast_cancer_wisconsin/train.csv")


print("Initializing Classifier...\n")
classifier = PCAClassifier(df=breast_cancer_wisconsin)


print("Computing principal components...\n")
classifier.compute_principal_components()

print("Printing principal components...\n")
for i, c in enumerate(classifier.classes):
    print(f"CLASS {i}")
    print(c.components)

print("Printing principal components explained variance...\n")
for i, c in enumerate(classifier.classes):
    print(f"CLASS {i}")
    print(c.explained_variances)

print("Computing principal components similaritys...\n")
classifier.compute_principal_components_similarity()

print("Printing principal components similarity matrix...\n")
for i, c in enumerate(classifier.classes):
    print(f"CLASS {i}")
    print(c.similarity_matrix)



print("Initializing Visualizer")
visualizer = PCAVisualizer(classifier.classes)

print("Ploting Cumulative Variances")
visualizer.cumulative_variance()

print("Ploting components similarity matrix")
visualizer.components_similarity()

