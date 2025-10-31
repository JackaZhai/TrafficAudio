# Import necessary libraries
import librosa
import soundfile as sf
import csv
import numpy as np

def extract_audio_segment(input_path, center_sec, range_sec, dest_path):
    """
    Extracts a stereo audio segment from an input audio file and saves it to a specified destination.
    
    Parameters:
    - input_path: Path to the input WAV file.
    - center_sec: The center point in seconds to extract the audio segment from.
    - range_sec: The range in seconds to extract around the center point.
    - dest_path: The destination file path to save the extracted segment.
    """
    center_sec = float(center_sec)
    center_sample = int(center_sec * sample_rate)
    range_samples = int(range_sec * sample_rate)
    
    # Determine start and end sample indices for the extraction
    start_sample = max(0, center_sample - range_samples)
    end_sample = min(len(Left_Channel), center_sample + range_samples)
    
    # Extract left and right channels
    audio_segment_left = Left_Channel[start_sample:end_sample]
    audio_segment_right = Right_Channel[start_sample:end_sample]
    
    # Stack the left and right channels into a stereo audio
    stereo_audio = np.vstack((audio_segment_left, audio_segment_right))
    
    # Save the audio segment to the destination path
    sf.write(dest_path, stereo_audio.T, sample_rate)

def process_csv_and_extract_segments(csv_file_path, wav_file_path, dest_path, range_sec=1):
    """
    Reads data from a CSV file and extracts audio segments based on the information in the file.

    Parameters:
    - csv_file_path: Path to the CSV file containing the segment information.
    - wav_file_path: Path to the input WAV file.
    - dest_path: Destination directory to save the extracted audio files.
    - range_sec: Range in seconds to extract around the center point (default is 1 second).
    """
    # Load the stereo waveform and sample rate from the input WAV file
    waveform, sample_rate = librosa.load(wav_file_path, sr=None, mono=False)
    global Left_Channel, Right_Channel
    Left_Channel, Right_Channel = waveform[0], waveform[1]

    row_count = 0
    
    # Open the CSV file
    with open(csv_file_path, 'r') as csv_file:
        csv_reader = csv.DictReader(csv_file)
        
        # Iterate through each row in the CSV
        for row in csv_reader:
            # Construct the file name for the extracted segment
            file_name = construct_file_name(row, dest_path)
            
            # Calculate the center second for extraction
            center_sec = calculate_center_sec(row)
            
            # Extract the audio segment
            extract_audio_segment(wav_file_path, center_sec, range_sec, file_name)
            row_count += 1

    print(f"Total Rows Processed: {row_count}")

def construct_file_name(row, dest_path):
    """
    Constructs the destination file name based on the CSV row information.

    Parameters:
    - row: The row from the CSV file.
    - dest_path: The destination directory for saving the audio segment.

    Returns:
    - The constructed file name as a string.
    """
    file_name = dest_path + "\\" + "2023-08-03_" + format_time(row) + format_vehicle_info(row) + "2L.wav"
    return file_name

def format_time(row):
    """
    Formats the time information from the CSV row into a string.
    """
    timee = "15-" + str(int(row["Time_min"])) + "-" + str(float(row["Time_sec"])) + "-"
    return timee

def calculate_center_sec(row):
    """
    Calculates the center second for audio segment extraction from the CSV row information.
    """
    ssec = 60 * int(row["Time_min"])
    secs = float(row["Time_sec"])
    return ssec + secs

def format_vehicle_info(row):
    """
    Formats the vehicle type, direction, passing mode, and status from the CSV row into a string.
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


# Set file paths for processing
wav_file_path = r'C:\path_to_your_audio.wav'
csv_file_path = r'C:\path_to_your_csv.csv'
dest_path = r'C:\path_to_destination_directory'

# Process CSV and extract audio segments
process_csv_and_extract_segments(csv_file_path, wav_file_path, dest_path)
