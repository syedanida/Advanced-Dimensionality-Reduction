# Databricks notebook source
!curl -L -o archive.zip https://www.kaggle.com/api/v1/datasets/download/uciml/iris

# COMMAND ----------

!unzip -o archive.zip -d iris-dataset-cleaned

# COMMAND ----------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# COMMAND ----------

# Load dataset
iris_df = pd.read_csv('iris-dataset-cleaned/Iris.csv')

# COMMAND ----------

iris_features = iris_df.drop(['Id', 'Species'], axis=1)

# COMMAND ----------

scaler = StandardScaler()
iris_standardized = scaler.fit_transform(iris_features)

# COMMAND ----------

pca = PCA(n_components=2)  # Reduce to 2 dimensions
iris_pca = pca.fit_transform(iris_standardized)

# COMMAND ----------

print("Explained Variance Ratio:", pca.explained_variance_ratio_)

# COMMAND ----------

iris_df['PCA1'] = iris_pca[:, 0]
iris_df['PCA2'] = iris_pca[:, 1]

# COMMAND ----------

plt.figure(figsize=(8, 6))
for species in iris_df['Species'].unique():
    subset = iris_df[iris_df['Species'] == species]
    plt.scatter(subset['PCA1'], subset['PCA2'], label=species)

plt.title('PCA of Iris Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
