import numpy as np
import pandas as pd

df = pd.read_csv("breast_cancer_wisconsis.csv")

labels = df["diagnosis"]
df = df.drop(columns=["id", "diagnosis"])

# centralize and normalize data with z-score  # min max would corrupt (co)variance info

for column in df:
    df[column] = (df[column] - df[column].mean())/df[column].var()

# set up covariance matrix

cov_matrix_df = df.cov()
cov_matrix = cov_matrix_df.to_numpy()
print(cov_matrix)

# eigendecompose covariance matrix



# normalize eigenvectors
