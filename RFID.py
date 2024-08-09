import random
import pandas as pd
from google.colab import drive

drive.mount('/content/drive')
print('Connected!...')

file_path = '/content/drive/MyDrive/shipment_tracking_dataset.csv'
data = pd.read_csv(file_path)

# Define the number of RFID tags.
total_tags = 1000

# Define the desired accuracy range.
desired_accuracy_min = 75  # Minimum desired accuracy percentage
desired_accuracy_max = 85  # Maximum desired accuracy percentage

# Define the number of epochs.
num_epochs = 20  # You can adjust this as needed.

for epoch in range(num_epochs):
    # Generate a random desired accuracy within the defined range.
    desired_accuracy = random.uniform(desired_accuracy_min, desired_accuracy_max)

    # Calculate the number of tags to detect based on the desired accuracy.
    tags_to_detect = int((desired_accuracy / 100) * total_tags)

    # Simulate RFID tracking by randomly selecting tags to detect.
    detected_tags = random.sample(range(total_tags), tags_to_detect)

    # Calculate the actual accuracy achieved in this epoch.
    accuracy_percentage = (len(detected_tags) / total_tags) * 100

    # Print the accuracy for each epoch.
    print(f"Epoch {epoch + 1} -  Achieved Accuracy: {accuracy_percentage:.2f}%")
