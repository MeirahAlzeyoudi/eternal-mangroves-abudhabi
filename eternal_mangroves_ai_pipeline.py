# Eternal Mangroves: Discovering Abu Dhabi’s Forests – Past, Present, and Prospects
# Space Hackathon for Sustainability 2024 Winner - GreenWave Team

# Required Libraries
!pip install rasterio scikit-learn matplotlib seaborn numpy pillow

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import rasterio
from rasterio.plot import show
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import os
from PIL import Image

# Example NDVI image (replace with actual NDVI .tiff file)
ndvi_image_path = '/content/sample_ndvi_image.tiff'

# Step 1: Load NDVI Raster Image
def load_ndvi_image(path):
    with rasterio.open(path) as src:
        ndvi = src.read(1).astype('float32')
        profile = src.profile
    return ndvi, profile

ndvi_data, raster_profile = load_ndvi_image(ndvi_image_path)

# Step 2: Simulate label creation for training (e.g., healthy=3, medium=2, low=1, none=0)
def create_labels(ndvi):
    labels = np.zeros_like(ndvi, dtype=np.uint8)
    labels[ndvi <= 0] = 3          # Healthy Mangroves
    labels[(ndvi > 0) & (ndvi <= 0.3)] = 2  # Medium
    labels[(ndvi > 0.3) & (ndvi <= 0.6)] = 1  # Low
    labels[ndvi > 0.6] = 0         # No Mangrove
    return labels

labels_data = create_labels(ndvi_data)

# Step 3: Prepare Data for ML (Flatten image and labels)
x = ndvi_data.flatten().reshape(-1, 1)
y = labels_data.flatten()

# Step 4: Train-Test Split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

# Step 5: Train Random Forest Classifier
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Step 6: Evaluate the Model
y_pred = rf_model.predict(x_test)
print("Classification Report:\n", classification_report(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

# Step 7: Predict on Full NDVI Map
predicted_labels = rf_model.predict(x).reshape(ndvi_data.shape)

# Step 8: Visualize Prediction Map
plt.figure(figsize=(12, 8))
cmap = plt.cm.get_cmap('Greens', 4)
plt.imshow(predicted_labels, cmap=cmap, vmin=0, vmax=3)
plt.title('Predicted Mangrove Health Classification')
plt.colorbar(ticks=[0, 1, 2, 3], label='Health Category (0=No Mangrove, 3=Healthy)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.grid(False)
plt.tight_layout()
plt.show()

# Step 9: Calculate Area by Class
pixel_resolution = 10  # meters
pixel_area = pixel_resolution ** 2  # m^2 per pixel

area_summary = {
    'Healthy': np.sum(predicted_labels == 3) * pixel_area,
    'Medium': np.sum(predicted_labels == 2) * pixel_area,
    'Low': np.sum(predicted_labels == 1) * pixel_area,
    'No Mangrove': np.sum(predicted_labels == 0) * pixel_area
}

for category, area in area_summary.items():
    print(f"{category}: {area / 1000000:.2f} km²")

# Step 10: Estimate CO2 Sequestration (example logic)
healthy_area_km2 = area_summary['Healthy'] / 1e6
estimated_co2_per_km2 = 300000  # tonnes per km^2/year

total_co2_sequestration = healthy_area_km2 * estimated_co2_per_km2
print(f"Estimated CO2 Sequestration by Healthy Mangroves: {total_co2_sequestration:,.0f} tonnes/year")

# Optional: Save Prediction Output as New Raster
output_path = '/content/predicted_mangrove_health.tif'
with rasterio.open(
    output_path, 'w',
    driver='GTiff',
    height=predicted_labels.shape[0],
    width=predicted_labels.shape[1],
    count=1,
    dtype=rasterio.uint8,
    crs=raster_profile['crs'],
    transform=raster_profile['transform']
) as dst:
    dst.write(predicted_labels.astype(rasterio.uint8), 1)

print(f"Prediction raster saved at: {output_path}")
