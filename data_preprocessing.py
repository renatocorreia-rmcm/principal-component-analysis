import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

"""
    BREAST CANCER WISCONSIS
"""

dataset_name = "breast_cancer_wisconsis"

df = pd.read_csv(f"data/{dataset_name}/{dataset_name}.csv")

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

df_scaled.to_csv(f"data/{dataset_name}/{dataset_name}_processed.csv")
df_scaled_train.to_csv(f"data/{dataset_name}/{dataset_name}_processed_train.csv")
df_scaled_test.to_csv(f"data/{dataset_name}/{dataset_name}_processed_test.csv")
