##working code to inject noise to all the .h5py files stored in psuedocodes structure

import pandas as pd
import os
import h5py
import numpy as np
import cv2

Gestures = ['Gesture_0','Gesture_1','Gesture_2','Gesture_3','Gesture_4','Gesture_5',
            'Gesture_6','Gesture_7','Gesture_8','Gesture_9','Gesture_10','Gesture_11']

df = pd.read_csv("/home/shefali/Documents/Declaration_of_consent/Data_recording_codes.csv")
psuedo_codes = list(df['Anonymous code']) #list of codes assigned to the volunteers while collecting data

source_path = "/home/shefali/recorded_data_BB_reduced_size"
destination_path = "/home/shefali/recorded_data_BB_reduced_size_noise_injected"

#selection of mean and standard deviation for intensity of noise
mean = 65
stddev = 10

for code in psuedo_codes:
    dirname = os.path.join(source_path,code)
    dest_dirname = os.path.join(destination_path,code)
    listfile = os.listdir(dirname)
    listfile = sorted(listfile,key = str.casefold)
    for gesture in listfile:
        source_gesture_path = os.path.join(dirname,gesture)
        dest_gesture_path = os.path.join(dest_dirname,gesture)
        filenames = os.listdir(source_gesture_path)
        for i in filenames:
            source_file_path = os.path.join(source_gesture_path,i)
            destination_file_path = os.path.join(dest_gesture_path,i)
            source_file = h5py.File(source_file_path,'r')
            destination_file = h5py.File(destination_file_path,'a')
            for key in list(destination_file.keys()):
                # Access the dataset containing pixel values
                dataset = source_file[key]
                updated_dataset = destination_file[key]
                # Add Gaussian noise to the pixel values
                noisy_pixel_values = dataset[:] + np.random.normal(mean, stddev, dataset.shape)
                # Update the dataset with noisy pixel values
                updated_dataset[:] = noisy_pixel_values
            print(destination_file,'updated')
            #close the .h5py file
            destination_file.close()
            source_file.close()
        print(f'The files for gesture {gesture} is updated')
    print(f'Files for code {code} updated')
print('ALL FILES INJECTED WITH GAUSSIAN NOISE')
            
