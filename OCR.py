import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Generate synthetic GPS data (latitude and longitude) as features
np.random.seed(0)
num_samples = 1000
latitude = np.random.uniform(35.0, 45.0, num_samples)  # Simulated latitude values
longitude = np.random.uniform(-120.0, -70.0, num_samples)  # Simulated longitude values

# Generate synthetic target variable (altitude) based on a linear relationship with latitude and longitude
altitude = 100 * latitude + 50 * longitude + np.random.normal(0, 10, num_samples)

# Combine features and target variable
X_gps = np.column_stack((latitude, longitude))
y_gps = altitude

# Split the GPS data into training and testing sets
X_train_gps, X_test_gps, y_train_gps, y_test_gps = train_test_split(X_gps, y_gps, test_size=0.2, random_state=42)

# Lists to store the training MSE and validation MSE for each epoch
train_mse_gps = []
val_mse_gps = []

# Train the model for each epoch and track the MSE
max_epochs_gps = 20  # Define the maximum number of epochs

for epoch_gps in range(1, max_epochs_gps + 1):
    # Initialize the Random Forest regressor with a fixed number of estimators
    rf_model_gps = RandomForestRegressor(n_estimators=100, random_state=42)

    # Train the model with GPS data
    rf_model_gps.fit(X_train_gps, y_train_gps)

    # Calculate training MSE for Random Forest
    train_mse = np.mean((rf_model_gps.predict(X_train_gps) - y_train_gps) ** 2)

    # Calculate validation MSE for Random Forest
    val_mse = np.mean((rf_model_gps.predict(X_test_gps) - y_test_gps) ** 2)

    train_mse_gps.append(train_mse)
    val_mse_gps.append(val_mse)

    # Print the MSE for the current epoch
    print(f'Epoch {epoch_gps} -  Training MSE: {train_mse:.2f}, Validation MSE: {val_mse:.2f}')

# Plot the MSE for training and validation over the epochs
epochs_list_gps = range(1, max_epochs_gps + 1)
plt.figure(figsize=(10, 6))
plt.plot(epochs_list_gps, train_mse_gps, label='Training MSE', marker='o')
plt.plot(epochs_list_gps, val_mse_gps, label='Validation MSE', marker='o')
plt.xlabel('Epochs')
plt.ylabel('Mean Squared Error (MSE)')
plt.title('MSE vs. Epochs for GPS-Based Random Forest Regression')
plt.legend()
plt.grid(True)
plt.show()
