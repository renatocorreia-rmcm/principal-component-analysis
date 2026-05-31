import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import kagglehub


"""
    BREAST CANCER WISCONSIS
"""

dataset_name = "breast_cancer_wisconsin"
path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data", path="data.csv", output_dir=f"data/{dataset_name}")

df = pd.read_csv(path)

# drop indices and last void column
df = df.drop(columns=df.columns[[0, -1]].to_list())


targets = df["diagnosis"]
features = df.drop(columns=["diagnosis"])

targets = targets.map({'B': 0, 'M' : 1})

scaler = StandardScaler()  # z score scalling
features_scaled = pd.DataFrame(
    scaler.fit_transform(features),
    columns=features.columns
)

df_scaled = pd.concat([targets, features_scaled], axis=1)


df_scaled_train, df_scaled_test = train_test_split(df_scaled)

df_scaled.to_csv(f"data/{dataset_name}/processed.csv")
df_scaled_train.to_csv(f"data/{dataset_name}/train.csv")
df_scaled_test.to_csv(f"data/{dataset_name}/test.csv")


"""
    FASHION MNIST
"""

dataset_name = "fashion_mnist"

kagglehub.dataset_download("zalando-research/fashionmnist", path="fashion-mnist_train.csv", output_dir=f"data/{dataset_name}")
kagglehub.dataset_download("zalando-research/fashionmnist", path="fashion-mnist_test.csv", output_dir=f"data/{dataset_name}")

