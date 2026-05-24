import pandas as pd
from sklearn.preprocessing import StandardScaler

"""
    BREAST CANCER WISCONSIS
"""

dataset_name = "breast_cancer_wisconsis"

df = pd.read_csv(f"data/{dataset_name}.csv")

labels = df["diagnosis"]
numerical_columns_indices = df.columns.difference(["diagnosis"])
data = df[numerical_columns_indices]

scaler = StandardScaler()  # z score scalling
df_scaled = pd.DataFrame(
    scaler.fit_transform(data),
    columns=data.columns
)

df = pd.concat([labels, df_scaled], axis=1)

df.to_csv(f"data/{dataset_name}_processed.csv")