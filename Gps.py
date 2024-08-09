#GPS
import math
import random
import pandas as pd
from google.colab import drive
drive.mount('/content/drive')

print('Connected!...')

file_path = '/content/drive/MyDrive/shipment_tracking_dataset.csv'
data = pd.read_csv(file_path)


# Function to generate GPS coordinates with accuracy
def generate_gps_coordinates(reference_lat, reference_lon, accuracy_meters):
    earth_radius = 6371000  # Earth's radius in meters
    max_distance = accuracy_meters / earth_radius

    # Generate random bearing and distance
    random_bearing = random.uniform(0, 360)
    random_distance = max_distance  # Set the distance to the desired accuracy

    # Calculate new latitude and longitude
    reference_lat_rad = math.radians(reference_lat)
    reference_lon_rad = math.radians(reference_lon)

    new_lat_rad = math.asin(
        math.sin(reference_lat_rad) * math.cos(random_distance) +
        math.cos(reference_lat_rad) * math.sin(random_distance) * math.cos(math.radians(random_bearing))
    )

    new_lon_rad = reference_lon_rad + math.atan2(
        math.sin(math.radians(random_bearing)) * math.sin(random_distance) * math.cos(reference_lat_rad),
        math.cos(random_distance) - math.sin(reference_lat_rad) * math.sin(new_lat_rad)
    )

    new_lat = math.degrees(new_lat_rad)
    new_lon = math.degrees(new_lon_rad)

    return round(new_lat, 6), round(new_lon, 6)

# Function to calculate GPS accuracy
def calculate_gps_accuracy(reference_lat, reference_lon, generated_lat, generated_lon):
    earth_radius = 6371000  # Earth's radius in meters

    # Calculate the Haversine distance between the reference and generated points
    reference_lat_rad = math.radians(reference_lat)
    generated_lat_rad = math.radians(generated_lat)
    delta_lat_rad = generated_lat_rad - reference_lat_rad
    delta_lon_rad = math.radians(generated_lon - reference_lon)

    a = math.sin(delta_lat_rad / 2) ** 2 + math.cos(reference_lat_rad) * math.cos(generated_lat_rad) * math.sin(delta_lon_rad / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    distance = earth_radius * c

    return round(distance, 2)  # Round the distance to 2 decimal places for accuracy

# Example usage
reference_latitude = 37.7749  # Example reference latitude (San Francisco)
reference_longitude = -122.4194  # Example reference longitude (San Francisco)
desired_accuracy = 93 # Desired accuracy fixed at 80% (set as desired)
num_samples = 1 # Number of samples

for _ in range(num_samples):
    # Generate random GPS coordinates around the reference point with the desired accuracy
    generated_latitude, generated_longitude = generate_gps_coordinates(reference_latitude, reference_longitude, desired_accuracy)

    # Calculate and print the accuracy for each sample
    accuracy = calculate_gps_accuracy(reference_latitude, reference_longitude, generated_latitude, generated_longitude)
    print(f" Accuracy: {accuracy} ")
