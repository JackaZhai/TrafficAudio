#      LMS from all audio samples


import matplotlib.pyplot as plt
import IPython.display as ipd
import librosa
import librosa.display
import os
import numpy as np
import librosa
import imageio



##   Files in BG
BG_path = r'C:\_DS\_A_MYDS\BG\BG1' + "\\"

##   Files in Car
RL_path = r'C:\_DS\_A_MYDS\RL' + "\\"

##   Files in Motorcycle

 
### Function
def plot_spectrogram(Y, sr, hop_length, y_axis="linear"):
    audio_length = Y.shape[1] * hop_length / sr
    print(audio_length)
    plt.figure(figsize=(30, 10))
    librosa.display.specshow(Y, 
                             sr=sr, 
                             hop_length=hop_length, 
                             x_axis="s", 
                             y_axis=y_axis)
    plt.colorbar(format="%+2.f")
    plt.xticks(np.arange(0, audio_length, 1))
    plt.xlabel("Time (s)")

def save_spectrogram(Y, sr, hop_length, directory, y_axis="log"):
    plt.figure(figsize=(5, 3))
    librosa.display.specshow(Y,
                             sr=sr,
                             hop_length=hop_length,
                             x_axis=None,
                             y_axis=y_axis)
    plt.axis('off')
    # print(filename)
    plt.savefig(directory, bbox_inches='tight', pad_inches=0)
    plt.close()
    

# def save_spectrogram(Y, sr, hop_length, directory, y_axis="log"):
#     plt.figure(figsize=(5, 3))
#     librosa.display.specshow(Y,
#                              sr=sr,
#                              hop_length=hop_length,
#                              x_axis=None,
#                              y_axis=y_axis)
#     plt.axis('off')
#     # print(filename)
#     plt.savefig(directory, bbox_inches='tight', pad_inches=0)
#     plt.close()
def normalize(array):
    return 2 * ((array - np.min(array)) / np.ptp(array)) - 1

#%%
### Function
Pic_pathpath1 = r"C:\_DS\_A_MYDS\_LSPEC\RL"+"//"

#### Show the name of files in the folder
for filename in os.listdir(RL_path):
    FFF_name = filename
    file_path = os.path.join(RL_path, filename)
    if os.path.isfile(file_path):        
        # file is read and is in "filname"
        filename1 = RL_path + filename
        # SR = 10000
        FRAME_SIZE = 8192
        HOP_SIZE = 1048         
        # STFT
        audio_car_Unnormed1, sample_rate = librosa.load(filename1, sr=None)
        
        # audio_car = librosa.util.normalize(audio_car_Unnormed1)
        audio_car =audio_car_Unnormed1

        FFT_audio = librosa.stft(audio_car, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        Y_car = np.abs(FFT_audio) ** 2 
        # Y_car =  FFT_audio                     
        # Calculate Log-Amplitude Spectrogram
        Y_log_scale = librosa.power_to_db(Y_car)  
        # Y_log_scale = librosa.amplitude_to_db(Y_car) 
        
        # Set directory and filename
        directory = Pic_pathpath1 + "\\" + FFF_name + "_x4_.png"
        # Plot and save Log-Amplitude Spectrogram
        save_spectrogram(Y_log_scale, sample_rate, HOP_SIZE, directory)  


#%%
### Function
Pic_pathpath1 = r"C:\_DS\_A_MYDS\_LSPEC\BG"+"//"

i = 0
#### Show the name of files in the folder
for filename in os.listdir(BG_path):
    FFF_name = filename
    file_path = os.path.join(BG_path, filename)
    if os.path.isfile(file_path):        
        # file is read and is in "filname"
        filename1 = BG_path + filename
        # SR = 10000
        # FRAME_SIZE = 2048
        # HOP_SIZE = 1024         
        FRAME_SIZE = 8192
        HOP_SIZE = 1024
        
        # STFT
        audio_car_Unnormed1, sample_rate = librosa.load(filename1, sr=None)
        
        # audio_car = librosa.util.normalize(audio_car_Unnormed1)
        audio_car =audio_car_Unnormed1

        FFT_audio = librosa.stft(audio_car, n_fft=FRAME_SIZE, hop_length=HOP_SIZE)
        Y_car = np.abs(FFT_audio) ** 2 
        # Y_car =  FFT_audio                     
        # Calculate Log-Amplitude Spectrogram
        Y_log_scale = librosa.power_to_db(Y_car)  
        # Y_log_scale = librosa.amplitude_to_db(Y_car) 
        
        # Set directory and filename
        directory = Pic_pathpath1 + "\\" + FFF_name + "_x2_.png"
        # Plot and save Log-Amplitude Spectrogram
        save_spectrogram(Y_log_scale, sample_rate, HOP_SIZE, directory)  
        i += 1
        if i% 25 ==0 :
            print(i)