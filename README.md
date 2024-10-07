# Project Name

A brief description of what your project does, its features, and its purpose.

## Table of Contents
1. [Installation](#installation)
2. [Vehicle Audio Sample Etraction](#contributing)
3. [Background Audio Sample Etraction](#contributing)
4. [Feature Extraction](#features)
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

## Vehicle Audio Sample Extraction
We use the code Veh_Maker.py to extract the audio samples of vehicles as shown in the image below. 
The code loads an audio file and a .csv file that contains the time stamps of passing vehicles and relevant data. Then it uses this information to find the location that should be cropped from the audio. 


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
