## to add random small translatons in an image stored in .h5py format


import os
import h5py
import cv2
import numpy as np
from scipy.ndimage import shift

Gestures = ['Gesture_0','Gesture_1','Gesture_2','Gesture_3','Gesture_4','Gesture_5',
            'Gesture_6','Gesture_7','Gesture_8','Gesture_9','Gesture_10','Gesture_11']

df = pd.read_csv("/home/shefali/Documents/Declaration_of_consent/Data_recording_codes.csv")
psuedo_codes = list(df['Anonymous code']) #list of codes assigned to the volunteers while collecting data

source_path = "/home/shefali/recorded_data_BB_reduced_size"
destination_path = "/home/shefali/recorded_data_BB_reduced_size_translated"

def add_translations(image, max_shift=25):
    # Generate random shift values
    shift_x = np.random.randint(-max_shift, max_shift + 1)
    shift_y = np.random.randint(-max_shift, max_shift + 1)
    # Apply horizontal and vertical translations
    translated_image = shift(image, (shift_y, shift_x, 0), mode='wrap')
    return translated_image


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
                dataset = source_file[key]
                updated_dataset = destination_file[key]
                translated_image = add_translations(dataset)
                # Update the dataset with noisy pixel values
                updated_dataset[:] = translated_image
            print(destination_file,'updated')
            #close the .h5py file
            destination_file.close()
            source_file.close()
        print(f'The files for gesture {gesture} is updated')
    print(f'Files for code {code} updated')
print('ALL FILES ADDED WITH RANDOM TRANSLATIONS')
            
    
    
