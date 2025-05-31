import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# Root directory of the renamed dataset
dataset_directory = 'Respiratory_Xray_Images'

# Updated class labels (previously: Covid19, Normal, Pneumonia)
image_classes = ['Virus_Infection', 'Healthy_Lungs', 'Bacterial_Infection']
class_indices = {label: index for index, label in enumerate(image_classes)}

# Define image size
image_size = (224, 224)

# Function to load and preprocess the dataset
def load_xray_images(data_root, labels, target_size=(224, 224)):
    image_data = []
    image_labels = []

    for label in labels:
        folder_path = os.path.join(data_root, label)
        class_index = class_indices[label]

        for filename in os.listdir(folder_path):
            file_path = os.path.join(folder_path, filename)
            image = cv2.imread(file_path)
            if image is None:
                continue  # Skip unreadable images
            image = cv2.resize(image, target_size)
            image_data.append(image)
            image_labels.append(class_index)

    # Convert to NumPy arrays
    X = np.array(image_data, dtype='float32') / 255.0
    y = to_categorical(np.array(image_labels), num_classes=len(labels))
    return X, y

# Load the images and labels
features, targets = load_xray_images(dataset_directory, image_classes, image_size)

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(features, targets, test_size=0.2, random_state=42)

print("Dataset loaded successfully!")
print("Training set:", X_train.shape, y_train.shape)
print("Test set:", X_test.shape, y_test.shape)
