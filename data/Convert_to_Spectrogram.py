# %% Imports
# Import libraries for audio processing, visualization, and file handling
import matplotlib.pyplot as plt
import librosa
import librosa.display
import os
import numpy as np

# %% Paths and Parameters
# Define paths for audio files and output directory
BG_path = r'PATH TO BACKGROUND AUDIO\BG_SW\\'  # Background audio files
Car_path = r'PATH TO MC AUDIO\MC\\'    # Vehicle audio files
Main_Pic_path = r"PATH TO SAVE LMS OF MC\MC_IMAGES\\"  # Directory to save spectrograms

# Parameters for audio processing
FRAME_SIZE = 512  # FFT window size
HOP_SIZE = 64     # Hop length for STFT
No_of_Samples = 5  # Number of samples to process

# %% Functions

def save_spectrogram(Y, sr, hop_length, save_path, y_axis="mel"):
    """
    Save a spectrogram as an image.
    
    Parameters:
        Y (numpy array): Spectrogram data
        sr (int): Sampling rate of the audio
        hop_length (int): Hop length for STFT
        save_path (str): File path to save the spectrogram
        y_axis (str): Scale for the y-axis ('linear', 'mel', etc.)
    """
    plt.figure(figsize=(5, 3))
    librosa.display.specshow(Y, sr=sr, hop_length=hop_length, y_axis=y_axis)
    plt.axis('off')  # Remove axes for cleaner visualization
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def normalize(audio):
    """
    Normalize an audio signal to the range [-1, 1].
    
    Parameters:
        audio (numpy array): Input audio signal
    
    Returns:
        numpy array: Normalized audio signal
    """
    return 2 * ((audio - np.min(audio)) / np.ptp(audio)) - 1

#%%
i = 0  # Counter for processed files
for filename in os.listdir(Car_path):
    if filename != 'desktop.ini':  # Skip system files
        file_path = os.path.join(Car_path, filename)
        
        if os.path.isfile(file_path):
            # Load and normalize the audio
            audio_car_unormed, sample_rate = librosa.load(file_path, sr=None)
            audio_car = normalize(audio_car_unormed)

            # Generate a mel-spectrogram and convert it to log scale
            mel_spectrogram = librosa.feature.melspectrogram(
                y=audio_car, sr=sample_rate, n_fft=FRAME_SIZE, hop_length=HOP_SIZE, n_mels=64
            )
            log_mels = librosa.power_to_db(mel_spectrogram, ref=np.max)

            # Set the filename for saving the spectrogram
            save_filename = f"{Main_Pic_path}{filename}__{FRAME_SIZE}-{HOP_SIZE}.png"
            
            # Save the spectrogram image
            save_spectrogram(log_mels, sample_rate, HOP_SIZE, save_filename)
            
            i += 1
            print(f"Processed file {i}: {filename}")

print(f"Total files processed: {i}")
