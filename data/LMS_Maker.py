import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import numpy as np

# Paths for audio files
BG_path = r'C:\_DS\_A_MYDS\_NEW_DS\BG_SW\\'
Car_path = r'C:\_DS\_MyDS\_VClasses\MC\\'

# Set the main directory for saving spectrograms
Main_Pic_path = r"C:\_DS\_MyDS\_TEST LMS\02_MC_TEST\\"

# Function to plot the spectrogram
def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    audio_length = Y.shape[1] * hop_length / sr
    print(f"Audio length: {audio_length} seconds")
    plt.figure(figsize=(30, 10))
    librosa.display.specshow(Y, sr=sr, hop_length=hop_length, x_axis="s", y_axis=y_axis)
    plt.colorbar(format="%+2.f dB")
    plt.xticks(np.arange(0, audio_length, 1))
    plt.xlabel("Time (s)")
    plt.show()

# Function to save the spectrogram as an image
def save_spectrogram(Y, sr, hop_length, save_path, y_axis="mel"):
    plt.figure(figsize=(5, 3))
    librosa.display.specshow(Y, sr=sr, hop_length=hop_length, y_axis=y_axis)
    plt.axis('off')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()

# Function to normalize an audio signal to [-1, 1]
def normalize(audio):
    return 2 * ((audio - np.min(audio)) / np.ptp(audio)) - 1

# Function to identify outliers in an array based on IQR
def find_outliers(arr, threshold=2.5):
    q1, q3 = np.percentile(arr, [25, 75])
    iqr = q3 - q1
    lower_bound = q1 - (iqr * threshold)
    upper_bound = q3 + (iqr * threshold)
    outlier_indices = np.where((arr < lower_bound) | (arr > upper_bound))[0]
    outlier_values = arr[outlier_indices]
    return outlier_indices, outlier_values

# Parameters
FRAME_SIZE = 512
HOP_SIZE = 64
No_of_Samples = 5

# Process each file in the Car directory and save the spectrograms
i = 0
for filename in os.listdir(Car_path):
    if filename != 'desktop.ini':  # Skip system files
        file_path = os.path.join(Car_path, filename)
        
        if os.path.isfile(file_path):
            # Load and normalize the audio
            audio_car_unormed, sample_rate = librosa.load(file_path, sr=None)
            audio_car = normalize(audio_car_unormed)

            # Create mel-spectrogram and convert to log scale
            mel_spectrogram = librosa.feature.melspectrogram(y=audio_car, sr=sample_rate, 
                                                             n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=64)
            log_mels = librosa.power_to_db(mel_spectrogram, ref=np.max)

            # Set the directory and filename for saving
            save_filename = f"{Main_Pic_path}{filename}__{FRAME_SIZE}-{HOP_SIZE}.png"
            
            # Save the Log-Amplitude Mel-Spectrogram
            save_spectrogram(log_mels, sample_rate, HOP_SIZE, save_filename)
            
            i += 1
            print(f"Processed file {i}: {filename}")

print(f"Total files processed: {i}")
