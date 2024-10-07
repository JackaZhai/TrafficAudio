# MELAUDIS Dataset

MELAUDIS is the first real-world acoustic vehicle dataset collected in multi-lane multi-vehicle urban roads with up to three lanes, and with shared or dedicated tram lanes, bicycle and car park lanes. The dataset is explained in "MELAUDIS: A Large-Scale Benchmark Acoustic Dataset For Intelligent Transportation Systems Research" submitted to Nature- Scientific Data.

## Table of Contents
1. [Installation](#installation)
2. [Vehicle Audio Sample Extraction](#vehicle-audio-sample-extraction)
3. [Background Audio Sample Extraction](#background-audio-sample-extraction)
4. [Feature Extraction](#feature-extraction)
5. [License](#license)
6. [Contact](#contact)

## Installation

Provide step-by-step instructions on how to get the project up and running.
```bash
# Clone the repository
git clone https://github.com/parineh/MELAUDIS.git

# Navigate to the project directory
cd MELAUDIS

# Install dependencies (example for Python projects)
pip install -r requirements.txt
```



![Vehicles_ALL](https://github.com/user-attachments/assets/eb058fd7-dc2c-4abf-9d74-2beceb35fa19)


## Vehicle Audio Sample Extraction
This Python code "Veh_Maker.py" (available in the repository) is designed to extract vehicle audio segments from a stereo WAV file based on information provided in a CSV file. Each row in the CSV contains details such as the time (centered in minutes and seconds) around which the audio should be extracted, along with additional metadata like vehicle type, direction, and status. The script reads this information and then extracts and saves audio segments from the WAV file as new stereo audio files. The filenames for these files are dynamically constructed from the metadata, allowing easy identification and organization.

The script utilizes several key libraries: librosa for loading and manipulating the audio files, soundfile for saving the extracted audio segments, csv for reading the CSV file containing segment extraction information, and numpy for handling numerical operations like stacking the left and right audio channels.

The core functionality is divided into several functions. The extract_audio_segment() function performs the actual extraction of the audio segment from the stereo WAV file. Given the center time and range (duration before and after the center), it calculates the start and end sample positions based on the sample rate of the audio. It then extracts the relevant portions of both the left and right channels, stacks them to form a stereo audio segment, and saves the result to a specified file path.

The process_csv_and_extract_segments() function manages the overall workflow. It reads the CSV file and iterates through each row. For each row, it calculates the center time in seconds by converting the minutes and seconds provided in the CSV into total seconds. This function also constructs a meaningful filename for the extracted audio segment by combining the metadata like vehicle type, direction, and status, and then calls the extract_audio_segment() function to handle the extraction and saving of the audio segment.

Helper functions like construct_file_name(), calculate_center_sec(), and format_vehicle_info() streamline specific tasks. The construct_file_name() function generates the file name for the extracted audio segment using the time and metadata from the CSV file, ensuring that each segment is saved with a descriptive and meaningful filename, such as "2023-08-03_15-10-20_CAR_RL_Mov_2L.wav". The calculate_center_sec() function converts the time from the CSV row (in minutes and seconds) into total seconds, which is used to determine the center point of the audio segment. The format_vehicle_info() function formats metadata like vehicle type, direction, passing mode, and status into a descriptive string, which is incorporated into the filename.

The script begins by defining the file paths for the input WAV file, the CSV containing the segment information, and the destination folder for the extracted audio segments. It then calls the process_csv_and_extract_segments() function, which processes the CSV file row by row and extracts the corresponding audio segments. Each extracted segment is saved with a filename that reflects the context of the audio, such as vehicle type, direction, and status.

This script is particularly useful for projects that involve audio event detection, such as acoustic analysis in traffic studies. By extracting and saving specific audio segments from a longer recording, the script enables researchers or analysts to focus on relevant sections of the audio, like the sounds of passing vehicles or traffic events. The descriptive filenames make it easy to manage and categorize the extracted segments based on the metadata provided in the CSV file.

![Data_Anno_Conv4](https://github.com/user-attachments/assets/7c7b012d-5956-4b78-9100-84a709575293)

## Background Audio Sample Extraction
This Python script is designed to extract specific audio segments from a stereo WAV file using information stored in a CSV file. The CSV contains metadata such as the time (in minutes and seconds) at which each audio segment should be centered, as well as details like vehicle type, direction, and status. Each row in the CSV corresponds to an audio segment to be extracted from the original WAV file.

The script starts by importing essential libraries: librosa for handling audio loading and manipulation, soundfile for saving extracted audio segments, csv for reading the metadata file, and numpy for numerical operations.

The core functionality is defined in several functions:

1- extract_audio_segment(): This function takes in the center time (in seconds) and range (duration) for the segment and calculates the start and end sample indices for both the left and right stereo channels. The extracted segment is saved as a new stereo file in the specified output path. The sample indices are calculated based on the sample rate of the audio file to ensure precise timing.

2- process_csv_and_extract_segments(): This function manages the overall process by reading the CSV file and iterating through its rows. For each row, it calculates the center time (by converting the minutes and seconds to total seconds) and constructs a meaningful filename that includes details like vehicle type and direction. It then calls the extract_audio_segment() function to handle the actual extraction and saving of the segment

3- construct_file_name(): This function generates a descriptive filename for each extracted segment by combining the recording time, vehicle type, direction, and other metadata from the CSV. This ensures that each saved file has a meaningful and organized name.

4- format_vehicle_info() and format_time(): These helper functions format the vehicle-related data (like vehicle type and direction) and time from the CSV row into strings that are used to create the filename for each segment.

Once the paths to the audio file, CSV file, and destination directory are defined, the script processes the CSV and extracts the corresponding audio segments, saving them with appropriately constructed filenames. Each audio segment is centered around a specific time point and includes a specified range of audio samples before and after the center.

This script is particularly useful for applications like acoustic analysis, where precise audio segments need to be isolated from a larger recording, such as detecting and analyzing vehicle sounds in traffic environments. The filenames generated for each segment make it easy to identify the extracted audio based on its characteristics.


## Feature Extraction
This Python script "BG_Maker.py" (available in the repository) is designed to process audio files from different categories (e.g., car and background sounds), generate their Mel-Spectrograms, and save these spectrograms as images for further analysis. The Mel-Spectrogram represents the energy distribution of the audio signal over time across various frequency bands. This visualization is commonly used in machine learning and audio signal processing tasks, as it provides a useful representation of the frequency content of the audio.

The script starts by importing the necessary libraries: librosa for handling audio files and generating the spectrogram, matplotlib for plotting and saving the spectrograms, and numpy for numerical operations. It also sets up the file paths for the directories containing the audio files and the directory where the resulting spectrogram images will be saved.

The script contains several key functions:

1- plot_spectrogram(): This function plots the spectrogram of the audio file, displaying the time on the x-axis and the frequency on the y-axis. It calculates the length of the audio in seconds and adds color to the plot to represent amplitude

2- save_spectrogram(): This function saves the spectrogram as an image file without displaying it. It is useful for generating images in batch mode without needing to display them on-screen. The spectrogram is saved as a clean image with no axis labels or borders to optimize space and clarity.

3- normalize(): This function normalizes the audio signal to a range between -1 and 1. Normalization is important to ensure that all audio files are scaled consistently before creating the spectrograms, especially if the audio files have different amplitude ranges

4- find_outliers(): Although not used in the main execution, this function is designed to identify outliers in the data based on the Interquartile Range (IQR) method. This could be useful for identifying unusual or extreme values in the audio signal, potentially indicating noise or anomalies.

The script then defines important parameters such as FRAME_SIZE (which determines the size of the window used for the Short-Time Fourier Transform (STFT)) and HOP_SIZE (which defines the step size between successive windows). These parameters impact the resolution and appearance of the Mel-Spectrogram.

In the main processing loop, the script iterates through each audio file in the Car_path directory. It skips any system files like 'desktop.ini'. For each audio file, the script loads the audio data using librosa, normalizes the signal, and generates a Mel-Spectrogram. This spectrogram is then converted to a logarithmic decibel scale, which improves the visibility of quieter frequency components. The resulting Mel-Spectrogram is saved as a PNG image with a descriptive filename that includes information about the frame and hop sizes used in the spectrogram calculation.

After processing each file, the script prints the number of files processed and the filename of each spectrogram. Once all files are processed, the script outputs the total number of files that were processed.

This script is particularly useful for generating large datasets of spectrogram images, which can later be used for machine learning tasks like classification or clustering. By visualizing the frequency components of audio signals over time, researchers can extract valuable features from audio data and use them to train models for tasks such as vehicle sound classification, acoustic scene analysis, or anomaly detection.


## Licens

Provide step-by-step inst


## contact

PLease feel free to contact me for more information:
sean.parineh@gmail.com
