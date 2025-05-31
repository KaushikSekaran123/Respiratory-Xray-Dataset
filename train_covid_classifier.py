import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG19
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

# Renamed dataset path and labels
dataset_path = 'Respiratory_Xray_Images'
label_names = ['Virus_Infection', 'Healthy_Lungs', 'Bacterial_Infection']
label_map = {label: idx for idx, label in enumerate(label_names)}

# Image settings
image_size = (224, 224)
num_classes = len(label_names)

# Function to load images
def load_images(data_dir, labels, target_size):
    data, targets = [], []
    for label in labels:
        folder = os.path.join(data_dir, label)
        class_id = label_map[label]
        for fname in os.listdir(folder):
            path = os.path.join(folder, fname)
            img = cv2.imread(path)
            if img is None:
                continue
            img = cv2.resize(img, target_size)
            data.append(img)
            targets.append(class_id)
    return np.array(data), np.array(targets)

# Load data
X, y = load_images(dataset_path, label_names, image_size)
X = X.astype('float32') / 255.0
y = to_categorical(y, num_classes)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
augmentor = ImageDataGenerator(rotation_range=20,
                               zoom_range=0.2,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               horizontal_flip=True)

# Load VGG19 base
base_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
for layer in base_model.layers:
    layer.trainable = False

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
x = Dropout(0.25)(x)
output = Dense(num_classes, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)
model.compile(optimizer=Adam(learning_rate=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(augmentor.flow(X_train, y_train, batch_size=32),
                    validation_data=(X_test, y_test),
                    epochs=10,
                    steps_per_epoch=len(X_train) // 32)

# Evaluate model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test accuracy: {accuracy:.2f}")
