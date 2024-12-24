import os

def search_files_by_strings(directory, search_strings):
    """
    Searches for files containing all specified strings in their names, including in subdirectories.

    Parameters:
    - directory (str): The path to the main directory.
    - search_strings (list): A list of strings to search for in the file names.

    Returns:
    - List of file paths matching all the specified strings.
    """
    matching_files = []


    # Walk through the main directory and all its subdirectories
    for root, _, files in os.walk(directory):
        for file in files:
            if file.endswith(".wav"):  # Process only .wav files (adjust if needed)
                # Check if all search strings exist in the file name
                if all(search_string in file for search_string in search_strings):
                    matching_files.append(os.path.join(root, file))  # Save full file path
                    

    return matching_files


# Example usage
if __name__ == "__main__":
    directory_path = r"C:\_DS\_A_MYDS\MultiVehicle Analysis\1V"
    
    # User-specified search strings
    search_strings = ["Swanston9","FF", "Car", "1V"]  # Example: search for Swanston9 street, FF traffic, Car, 1V
    
    count = 0
    
    # Search for matching files
    results = search_files_by_strings(directory_path, search_strings)
    
    # Print results
    print("Matching files:")
    for file in results:
        print(file)
        count += 1
    print(f"All counts:{count}")

