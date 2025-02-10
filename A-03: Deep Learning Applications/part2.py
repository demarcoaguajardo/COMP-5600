import os
import kagglehub
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

import matplotlib.pyplot as plt

from sklearn.metrics import classification_report


# ---------- DOWNLOAD AND LOAD DATASET ----------

# Download dataset
path = kagglehub.dataset_download("puneet6060/intel-image-classification")
print("Path to dataset:", path)

# Define paths
trainPath = os.path.join(path, 'seg_train', 'seg_train')
valPath = os.path.join(path, 'seg_test', 'seg_test')

# Define image and batch size
imgSize = (150, 150)
batchSize = 32

# Create ImageDataGenerator for training and validating sets
trainDataGen = ImageDataGenerator(rescale=1./255)
valDataGen = ImageDataGenerator(rescale=1./255)

# Load images from directories
trainGen = trainDataGen.flow_from_directory(
    trainPath,
    target_size=imgSize,
    batch_size=batchSize,
    class_mode='categorical'
)

valGen = valDataGen.flow_from_directory(
    valPath,
    target_size=imgSize,
    batch_size=batchSize,
    class_mode='categorical'
)

# ---------- IMPLEMENT CNN MODEL 1 ----------
# 3 Conv Layers
# 3 Max Pooling Layers
# Dropout Layer and Dense Layer before Output Layer

# Build CNN Model 1

cnnModel1 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dropout(0.5),
    Dense(512, activation='relu'),
    Dense(6, activation='softmax')
])

# Compile CNN Model 1
cnnModel1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train CNN Model 1
history1 = cnnModel1.fit(trainGen, validation_data=valGen, epochs=20)


# ---------- IMPLEMENT CNN MODEL 2 ----------
# 6 Conv Layers
# 3 Max Pooling Layers
# Dropout Layer and Dense Layer before Output Layer


# Build CNN Model 2

cnnModel2 = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    Conv2D(256, (3, 3), activation='relu'),
    Flatten(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(6, activation='softmax')
])

# Compile CNN Model 2
cnnModel2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train CNN Model 2
history2 = cnnModel2.fit(trainGen, validation_data=valGen, epochs=20)

# ---------- REPORT RESULTS ----------

# Evaluate CNN Model 1
valLoss1, valAccuracy1 = cnnModel1.evaluate(valGen)
print(f'CNN Model 1 - Validation Loss: {valLoss1}, Validation Accuracy: {valAccuracy1}')

# Evaluate CNN Model 2
valLoss2, valAccuracy2 = cnnModel2.evaluate(valGen)
print(f'CNN Model 2 - Validation Loss: {valLoss2}, Validation Accuracy: {valAccuracy2}')

# ---------- PLOT RESULTS ----------

# Get batch of validation images
valImages, valLabels = next(valGen)

# Predict labels
predictions1 = cnnModel1.predict(valImages)
predictions2 = cnnModel2.predict(valImages)

# Visualize 2 samples from predictions 1
for i in range(2):
    plt.imshow(valImages[i])
    plt.title(f'Model 1 - Predicted: {np.argmax(predictions1[i])}, Actual: {np.argmax(valLabels[i])}')
    plt.show()

# Visualize 2 samples from predictions 2
for i in range(2):
    plt.imshow(valImages[i])
    plt.title(f'Model 2 - Predicted: {np.argmax(predictions2[i])}, Actual: {np.argmax(valLabels[i])}')
    plt.show()

# ---------- CLASSIFICATION REPORT ----------

# Evaluate CNN Model 1
valLoss1, valAccuracy1 = cnnModel1.evaluate(valGen)
print(f'CNN Model 1 - Validation Loss: {valLoss1}, Validation Accuracy: {valAccuracy1}')

# Evaluate CNN Model 2
valLoss2, valAccuracy2 = cnnModel2.evaluate(valGen)
print(f'CNN Model 2 - Validation Loss: {valLoss2}, Validation Accuracy: {valAccuracy2}')

# Get true and predicted labels for validation set
valGen.reset()
trueLabels = valGen.classes

# Predictions
predictions1 = cnnModel1.predict(valGen, steps=valGen.samples // valGen.batch_size + 1)
predictions2 = cnnModel2.predict(valGen, steps=valGen.samples // valGen.batch_size + 1)

# Get predicted labels
predictedLabels1 = np.argmax(predictions1, axis=1)
predictedLabels2 = np.argmax(predictions2, axis=1)

# Calculate classification report for Model 1
report1 = classification_report(trueLabels, predictedLabels1, 
                                target_names=valGen.class_indices.keys(), output_dict=True)
perClassAccuracy1 = {class_name: report1[class_name]['recall'] for class_name
                     in report1.keys() if class_name != 'accuracy'}
totalAccuracy1 = report1['accuracy']

# Calculate classification report for Model 2
report2 = classification_report(trueLabels, predictedLabels2,
                                target_names=valGen.class_indices.keys(), output_dict=True)
perClassAccuracy2 = {class_name: report2[class_name]['recall'] for class_name
                     in report2.keys() if class_name != 'accuracy'}
totalAccuracy2 = report2['accuracy']

# Print total accuracy and per-class accuracy for Model 1
print("Model 1 - Per-Class Accuracy:")
for class_name, accuracy in perClassAccuracy1.items():
    print(f"{class_name}: {accuracy:.2f}")

# Print total accuracy and per-class accuracy for Model 2
print("Model 2 - Per-Class Accuracy:")
for class_name, accuracy in perClassAccuracy2.items():
    print(f"{class_name}: {accuracy:.2f}")


