import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

data = pd.read_csv('patient_data.csv')
features = ['age', 'bmi', 'blood_pressure', 'cholesterol', 'glucose']
scaler = StandardScaler()
normalized_data = scaler.fit_transform(data[features])

kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(normalized_data)

data['cluster'] = cluster_labels
cluster_map = {0: 'Normal', 1: 'Healthy', 2: 'Weak'}
data['cluster_label'] = data['cluster'].map(cluster_map)
print("Prithvi Kathuria,21BBS0158")
print(data[['age', 'bmi', 'blood_pressure', 'cholesterol', 'glucose', 'cluster_label']])

plt.figure(figsize=(10, 6))
scatter = plt.scatter(data['age'], data['bmi'], c=data['cluster'], cmap='viridis')
plt.xlabel('Age')
plt.ylabel('BMI')
plt.title('Patient Clusters')
plt.colorbar(scatter)
plt.show()