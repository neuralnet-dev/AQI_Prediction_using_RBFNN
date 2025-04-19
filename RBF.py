import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import mean_squared_error, r2_score
from scipy.spatial.distance import cdist

df = pd.read_csv("city_day.csv")
print(df.info())
data_info = {
    "shape": df.shape,
    "columns": df.columns.tolist(),
    "head": df.head(),
    "info": df.info(),
    "missing_values": df.isnull().sum()
}
print(data_info)

df = df.dropna(subset=["AQI"])
df = df.drop(columns=["Date", "AQI_Bucket"], axis=1)
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col] = df[col].fillna(df[col].median())
label_encoder = LabelEncoder()
df['City'] = label_encoder.fit_transform(df['City'])

plt.figure(figsize=(12, 10))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
plt.title("Correlation Heatmap of Pollutants and AQI")
plt.show()

features = ["PM2.5", "PM10", "NO", "NO2", "NOx", "NH3", "CO", "SO2", "O3", "Benzene", "Toluene", "Xylene"]
X = df[features].values
y = df["AQI"].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

y = y.reshape(-1, 1)
y_scaler = StandardScaler()
y_scaled = y_scaler.fit_transform(y).flatten()

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.3, random_state=42)

# Elbow method
clusters = [50, 100, 150, 200, 250, 500, 600, 700, 800, 900, 1000, 1500, 3000]
ssd = []

for k in clusters:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_train)
    ssd.append(kmeans.inertia_)

plt.figure(figsize=(8, 5))
plt.plot(clusters, ssd, marker='o', linestyle='-')
plt.xlabel("Number of Clusters")
plt.ylabel("Sum of Squared Distances (SSD)")
plt.title("Elbow Method for Optimal Clusters")
plt.show()

num_clusters = 500
kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(X_train)
centers = kmeans.cluster_centers_

pairwise_distances = cdist(centers, centers, 'euclidean')
sigma = np.median(pairwise_distances) / np.sqrt(2)

distances_train = cdist(X_train, centers, 'euclidean')
R_train = np.exp(- (distances_train ** 2) / (2 * sigma ** 2))
W = np.dot(np.linalg.pinv(R_train), y_train)

distances_test = cdist(X_test, centers, 'euclidean')
R_test = np.exp(- (distances_test ** 2) / (2 * sigma ** 2))

y_pred_train = np.dot(R_train, W)
y_pred_test = np.dot(R_test, W)

mse_train = mean_squared_error(y_train, y_pred_train)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_pred_train)

mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_test, y_pred_test)

print("\nTraining Performance (Normalized):")
print("MSE:", mse_train)
print("RMSE:", rmse_train)
print("R² Score:", r2_train)

print("\nTesting Performance (Normalized):")
print("MSE:", mse_test)
print("RMSE:", rmse_test)
print("R² Score:", r2_test)

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred_test, color='green', edgecolor='k', alpha=0.7)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', linewidth=2)
plt.title("Scatter Plot: Actual vs Predicted AQI (Normalized)")
plt.xlabel("Actual AQI (normalized)")
plt.ylabel("Predicted AQI (normalized)")
plt.grid(alpha=0.3)
plt.show()

subset_size = 100
indices = np.arange(subset_size)
y_test_subset = y_test[:subset_size]
y_pred_subset = y_pred_test[:subset_size]

plt.figure(figsize=(12, 6))
plt.plot(indices, y_test_subset, label='Actual AQI', color='red', linewidth=2)
plt.plot(indices, y_pred_subset, label='Predicted AQI', color='blue', linestyle='dashed', linewidth=2)
plt.title("Actual vs Predicted AQI (Subset, Normalized)")
plt.xlabel("Data Points")
plt.ylabel("AQI (normalized)")
plt.legend()
plt.grid(True)
plt.show()
