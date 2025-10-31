# %% Imports
#------------------------------------
#   Vehicle Detection - Method 2
#------------------------------------
# Import essential libraries for data handling, visualization, and machine learning
import numpy as np
import pandas as pd
import os
import cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.utils import shuffle
import matplotlib.pyplot as plt
from tqdm import tqdm

# %% Setup and Data Classes
# Define class labels for binary classification
class_names = ['BG', 'Veh']  # 'BG' represents background noise, 'Veh' represents vehicles
class_names_label = {class_name: i for i, class_name in enumerate(class_names)}

# Print the mapping of class names to labels
print("Class Names and Labels:", class_names_label)

# %% Data Loading Function
def load_data():
    """
    Load image data from a directory, convert it to RGB, resize, and store with labels.
    Returns:
        images: Numpy array of image data
        labels: Corresponding labels for the images
        file_names: File names for reference
    """
    DIRECTORY = r"Path to the directory of audio files\BG_and_Veh"

    images = []
    labels = []
    file_names = []

    for folder in os.listdir(DIRECTORY):
        if folder == 'desktop.ini':  # Skip system files
            continue

        label = class_names_label[folder]
        print(f"Folder: {folder}, Label: {label}")

        for file in os.listdir(os.path.join(DIRECTORY, folder)):
            img_path = os.path.join(DIRECTORY, folder, file)
            try:
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = cv2.resize(image, (223, 163))  # Resize image for uniform input size
                images.append(image)
                labels.append(label)
                file_names.append(file)
            except Exception as e:
                print(f"Error reading file: {img_path}\nError: {str(e)}")

    images = np.array(images, dtype='float32')
    labels = np.array(labels, dtype='int32')

    return images, labels, file_names

# %% Load and Shuffle Data
images, labels, file_names = load_data()
images, labels, file_names = shuffle(images, labels, file_names, random_state=22)

# Split data into train+validation and test sets
train_val_images, test_images, train_val_labels, test_labels, train_val_fnames, test_fnames = train_test_split(
    images, labels, file_names, test_size=0.2, random_state=11
)

# Further split train+validation data into train and validation sets
train_images, val_images, train_labels, val_labels, train_fnames, val_fnames = train_test_split(
    train_val_images, train_val_labels, train_val_fnames, test_size=0.1, random_state=11
)

# %% Display Class Distribution
def display_class_distribution(labels, dataset_type):
    """
    Display the number of samples per class in a dataset.
    """
    label_counts = np.bincount(labels)
    for i, count in enumerate(label_counts):
        print(f"{class_names[i]} - {dataset_type}: {count}")

print("Class distribution in datasets:")
display_class_distribution(train_labels, 'Train')
display_class_distribution(val_labels, 'Validation')
display_class_distribution(test_labels, 'Test')

# %% CNN Model Definition
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(163, 223, 3)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(64, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(128, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Conv2D(256, (3, 3), activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.MaxPooling2D(2, 2),

    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')  # Single output for binary classification
])

model.summary()

# %% Compile and Train the Model
def step_decay(epoch):
    """
    Learning rate schedule: Reduces learning rate by 10x every 25 epochs.
    """
    initial_lr = 0.0001
    drop = 0.1
    epochs_drop = 25
    lr = initial_lr * (drop ** (epoch // epochs_drop))
    return lr

learning_rate_scheduler = tf.keras.callbacks.LearningRateScheduler(step_decay, verbose=1)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

history = model.fit(
    train_images, train_labels, batch_size=128, epochs=10,
    validation_data=(val_images, val_labels), callbacks=[learning_rate_scheduler]
)

# %% Test and Evaluate the Model
# Make predictions on the test set
predictions = model.predict(test_images)
predictions = (predictions > 0.5).astype(int)

# Save classification results to a DataFrame
results = pd.DataFrame({
    'file_name': test_fnames,
    'actual_label': test_labels,
    'predicted_label': predictions.flatten(),
    'correct': test_labels == predictions.flatten()
})

# Save results to a CSV file
results.to_csv('classification_results.csv', index=False)

# Print summary of correct/incorrect predictions
correct_count = results['correct'].sum()
incorrect_count = len(results) - correct_count
print(f"Correctly classified: {correct_count}")
print(f"Incorrectly classified: {incorrect_count}")

# %% Confusion Matrix
conf_matrix = confusion_matrix(test_labels, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=conf_matrix, display_labels=class_names)

# Plot Confusion Matrix
fig, ax = plt.subplots(figsize=(8, 6))
disp.plot(cmap=plt.cm.Blues, ax=ax)
plt.title('Confusion Matrix')
plt.savefig('confusion_matrix.png', dpi=600)
plt.show()

# %% Plot Training History
def plot_training_history(history):
    """
    Plot accuracy and loss curves for training and validation.
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 5))

    # Accuracy Plot
    axs[0].plot(history.history['accuracy'], label='Train Accuracy')
    axs[0].plot(history.history['val_accuracy'], label='Validation Accuracy')
    axs[0].set_title('Accuracy')
    axs[0].set_xlabel('Epochs')
    axs[0].set_ylabel('Accuracy')
    axs[0].legend()

    # Loss Plot
    axs[1].plot(history.history['loss'], label='Train Loss')
    axs[1].plot(history.history['val_loss'], label='Validation Loss')
    axs[1].set_title('Loss')
    axs[1].set_xlabel('Epochs')
    axs[1].set_ylabel('Loss')
    axs[1].legend()

    plt.tight_layout()
    plt.savefig('training_history.png', dpi=600)
    plt.show()

plot_training_history(history)
