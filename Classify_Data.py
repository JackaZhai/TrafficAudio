# %% Imports  
#
#---------------------------------
#    Vehicle Detection
#---------------------------------
#
# Import essential libraries for data handling, visualization, and machine learning
import numpy as np
import pandas as pd
import os
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sn
import cv2
import tensorflow as tf
from tqdm import tqdm

# Set Seaborn font scale for better visualization
sn.set(font_scale=1.4)

# %% Data Classes
# Define class labels for binary classification
class_names = ['Veh', 'NoVeh']
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

# Print the mapping of class names to labels
print("Class Names and Labels:", class_names_label)

# %% Load Data Function
def load_data():
    """
    Load and preprocess image data from the specified directory.
    Organizes the data into training and testing sets with labels.

    Returns:
        output: A list containing (images, labels) tuples for Train and Test datasets.
    """
    # Define the dataset directory and categories
    DIRECTORY = r" \ Path to audio files\BG_and_Veh"
    CATEGORIES = ["Train", "Test"]

    output = []

    for category in CATEGORIES:
        path = os.path.join(DIRECTORY, category)
        print(f"Loading data from: {path}")

        images = []
        labels = []

        for folder in os.listdir(path):
            if folder == 'desktop.ini':  # Skip system files
                continue

            label = class_names_label[folder]
            print(f"Folder: {folder}, Label: {label}")

            for file in os.listdir(os.path.join(path, folder)):
                img_path = os.path.join(path, folder, file)
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                images.append(image)
                labels.append(label)

        images = np.array(images, dtype='float32')
        labels = np.array(labels, dtype='int32')
        output.append((images, labels))

    return output

# %% Load and Split Data
# Load the dataset
(train_images, train_labels), (test_images, test_labels) = load_data()

# Shuffle and split the dataset into training and validation sets
train_images, train_labels = shuffle(train_images, train_labels, random_state=11)
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# Print the sizes of the datasets
print(f"Train Data Size: {len(train_images)}")
print(f"Validation Data Size: {len(val_images)}")
print(f"Test Data Size: {len(test_images)}")

# %% Define the CNN Model
model = tf.keras.Sequential([
    # Layer 1
    tf.keras.layers.Conv2D(128, (3, 3), activation='relu', input_shape=(163, 279, 3)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Layer 2
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Layer 3
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling2D(2, 2),

    # Layer 4
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.2),

    # Layer 5
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.2),

    # Flatten and Fully Connected Layers
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# %% Compile the Model
learning_rate = 0.00001
optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# %% Train the Model
history = model.fit(
    train_images, train_labels, batch_size=128, epochs=100, validation_data=(val_images, val_labels)
)

# %% Evaluate the Model
test_loss, test_accuracy = model.evaluate(test_images, test_labels)
print(f"Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")

# %% Save the Model
save_weights = r"C:\_DS\_A_MYDS\_LSPEC\_Results\CNN-100ep\weights.h5"
save_model = r"C:\_DS\_A_MYDS\_LSPEC\_Results\CNN-100ep\entire_model.h5"
model.save_weights(save_weights)
model.save(save_model)

# %% Plot Accuracy and Loss
def plot_accuracy_loss(history):
    """
    Plot training and validation accuracy and loss over epochs.
    """
    fig = plt.figure(figsize=(20, 10))

    # Plot Accuracy
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], 'bo--', label="Train Accuracy")
    plt.plot(history.history['val_accuracy'], 'ro--', label="Validation Accuracy")
    plt.title("Accuracy: Train vs Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    # Plot Loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], 'bo--', label="Train Loss")
    plt.plot(history.history['val_loss'], 'ro--', label="Validation Loss")
    plt.title("Loss: Train vs Validation")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.show()

plot_accuracy_loss(history)

# %% Confusion Matrix
predictions = model.predict(test_images)
predicted_labels = (predictions > 0.5).astype(int)

# Compute Confusion Matrix
cm = confusion_matrix(test_labels, predicted_labels)
classes = ['Veh', 'NoVeh']

# Plot Confusion Matrix
plt.figure(figsize=(8, 6))
plt.imshow(cm, cmap=plt.cm.Blues)
plt.title('Confusion Matrix', fontsize=18)
plt.colorbar()
tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=45)
plt.yticks(tick_marks, classes)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')

# Annotate Confusion Matrix
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(j, i, format(cm[i, j], 'd'), ha="center", va="center",
                 color="white" if cm[i, j] > cm.max() / 2 else "black")

plt.tight_layout()
plt.show()
