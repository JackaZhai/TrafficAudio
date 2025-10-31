# Import necessary libraries
import librosa
import soundfile as sf
import csv
import numpy as np

def extract_audio_segment(center_sec, range_sec, sample_rate, left_channel, right_channel, dest_path):
    """
    Extracts a stereo audio segment from the left and right channels and saves it to the specified path.

    Parameters:
    - center_sec: The center point in seconds to extract the audio segment from.
    - range_sec: The range in seconds to extract around the center point.
    - sample_rate: Sample rate of the audio file.
    - left_channel: Left channel of the stereo audio.
    - right_channel: Right channel of the stereo audio.
    - dest_path: The destination file path to save the extracted segment.
    """
    center_sample = int(center_sec * sample_rate)
    range_samples = int(range_sec * sample_rate)
    
    start_sample = max(0, center_sample - range_samples)
    end_sample = min(len(left_channel), center_sample + range_samples)
    
    # Extract audio segments for both channels
    audio_segment_left = left_channel[start_sample:end_sample]
    audio_segment_right = right_channel[start_sample:end_sample]
    
    # Stack into stereo audio
    stereo_audio = np.vstack((audio_segment_left, audio_segment_right))
    
    # Save the stereo audio segment
    sf.write(dest_path, stereo_audio.T, sample_rate)

def process_csv_and_extract_segments(csv_file_path, wav_file_path, dest_path, range_sec=1):
    """
    Reads data from a CSV file and extracts audio segments from a WAV file.

    Parameters:
    - csv_file_path: Path to the CSV file containing segment information.
    - wav_file_path: Path to the input WAV file.
    - dest_path: Path to save the extracted audio segments.
    - range_sec: The range in seconds to extract around the center point.
    """
    # Load the stereo waveform and sample rate from the input WAV file
    waveform, sample_rate = librosa.load(wav_file_path, sr=None, mono=False)
    left_channel, right_channel = waveform[0], waveform[1]

    row_count = 0
    
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Iterate through each row in the CSV file
        for row in csv_reader:
            file_name = construct_file_name(row, dest_path)
            center_sec = calculate_center_sec(row)
            
            # Extract and save audio segment
            extract_audio_segment(center_sec, range_sec, sample_rate, left_channel, right_channel, file_name)
            row_count += 1
    
    print(f"Total Rows Processed: {row_count}")

def construct_file_name(row, dest_path):
    """
    Constructs the destination file name for the audio segment based on the CSV row data.

    Parameters:
    - row: The CSV row containing segment information.
    - dest_path: The destination directory for saving the audio segment.

    Returns:
    - The constructed file name.
    """
    time_str = f"15-{int(row['Time_min'])}-{float(row['Time_sec'])}-"
    vehicle_str = format_vehicle_info(row)
    file_name = f"{dest_path}\\2023-08-03_{time_str}Swanston_{vehicle_str}2L.wav"
    
    return file_name

def format_vehicle_info(row):
    """
    Formats the vehicle type, direction, passing mode, and status from the CSV row into a string.

    Parameters:
    - row: The CSV row containing segment information.

    Returns:
    - A formatted string based on vehicle information.
    """
    vehicle_type_map = {1: "BIC_", 2: "MC_", 3: "CAR_", 4: "VAN_", 5: "BUS_", 6: "TRC_", 7: "TRM_"}
    direction_map = {1: "RL_", 2: "LR_"}
    passing_mode_map = {1: "1V_", 2: "2V_", 3: "OppV_"}
    status_map = {1: "Mov_", 2: "Stat_"}

    vehicle_type = vehicle_type_map.get(int(row["Veh_Type"]), "")
    direction = direction_map.get(int(row["Direction"]), "")
    passing_mode = passing_mode_map.get(int(row["Pass_Mode"]), "")
    status = status_map.get(int(row["Status"]), "")

    return vehicle_type + direction + passing_mode + status

def calculate_center_sec(row):
    """
    Calculates the center second for extracting the audio segment.

    Parameters:
    - row: The CSV row containing time information.

    Returns:
    - The calculated center second.
    """
    ssec = 60 * int(row["Time_min"])
    secs = float(row["Time_sec"])
    return ssec + secs

# File paths
wav_file_path = r'...\Audio_file.wav'
csv_file_path = r'...\Annotations.csv'
dest_path = r'...\Destination_Folder'

# Process the CSV and extract audio segments
process_csv_and_extract_segments(csv_file_path, wav_file_path, dest_path)
