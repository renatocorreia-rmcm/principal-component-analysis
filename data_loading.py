import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import kagglehub
import os

# Create base data directory if it doesn't exist
os.makedirs("data/breast_cancer_wisconsin", exist_ok=True)
os.makedirs("data/fashion_mnist", exist_ok=True)

"""
    BREAST CANCER WISCONSIN
"""
dataset_name = "breast_cancer_wisconsin"

# Best practice: Download the whole dataset folder, kagglehub handles extraction automatically in its cache
bc_download_dir = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")

# Locate the file inside the downloaded directory
bc_path = os.path.join(bc_download_dir, "data.csv")
df = pd.read_csv(bc_path)

# drop indices and last void column
df = df.drop(columns=df.columns[[0, -1]].to_list())

targets = df["diagnosis"]
features = df.drop(columns=["diagnosis"])
targets = targets.map({'B': 0, 'M' : 1})

scaler = StandardScaler()
features_scaled = pd.DataFrame(
    scaler.fit_transform(features),
    columns=features.columns
)

df_scaled = pd.concat([targets, features_scaled], axis=1)
df_scaled_train, df_scaled_test = train_test_split(df_scaled)

# Save your processed outputs to your local data folder
df_scaled.to_csv(f"data/{dataset_name}/processed.csv", index=False)
df_scaled_train.to_csv(f"data/{dataset_name}/train.csv", index=False)
df_scaled_test.to_csv(f"data/{dataset_name}/test.csv", index=False)


"""
    FASHION MNIST
"""
dataset_name = "fashion_mnist"

# Instead of downloading twice, download the dataset ONCE.
# kagglehub will return the path to the unzipped folder containing both CSVs.
fashion_download_dir = kagglehub.dataset_download("zalando-research/fashionmnist")

# Point to the files inside the unzipped cache directory
train_path = os.path.join(fashion_download_dir, "fashion-mnist_train.csv")
test_path = os.path.join(fashion_download_dir, "fashion-mnist_test.csv")

fashion_train = pd.read_csv(train_path)
fashion_test = pd.read_csv(test_path)

# (Optional) Save them to your local data folder if you want a copy there
fashion_train.to_csv(f"data/{dataset_name}/train.csv", index=False)
fashion_test.to_csv(f"data/{dataset_name}/test.csv", index=False)

print("All datasets downloaded, processed, and saved successfully!")
