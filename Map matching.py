from geopy.distance import geodesic
import numpy as np
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

print('Connected!...')

file_path = '/content/drive/MyDrive/shipment_tracking_dataset.csv'
data = pd.read_csv(file_path)

# Sample GPS coordinates (latitude, longitude) and road network data
gps_data = [(40.748817, -73.985428), (40.749017, -73.985428), (40.749217, -73.985428)]
road_network = [(40.748827, -73.985428), (40.749027, -73.985428), (40.749227, -73.985428)]  # Sample road points

# Function to find the closest point on the road to a GPS point
def find_closest_point(gps_point, road_points):
    min_distance = float('inf')
    closest_point = None
    for road_point in road_points:
        distance = geodesic(gps_point, road_point).meters
        if distance < min_distance:
            min_distance = distance
            closest_point = road_point
    return closest_point, min_distance


# Perform map matching for each GPS point
matched_points = []
accuracy_values = []

# Define a reference distance for calculating accuracy percentage
reference_distance = 10.0  # Example: 10 meters

for gps_point in gps_data:
    closest_road_point, accuracy = find_closest_point(gps_point, road_network)
    matched_points.append(closest_road_point)

    # Calculate accuracy as a percentage
    accuracy_percentage = (1 - (accuracy / reference_distance)) * 100
    accuracy_values.append(accuracy_percentage)

# Calculate the average accuracy value as a percentage
average_accuracy_percentage = np.mean(accuracy_values)
print(f" Accuracy: {average_accuracy_percentage:.2f}%")
