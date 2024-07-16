##working code to add small random occlusions to all the .h5py files stored in psuedocodes structure

import os
import h5py
import numpy as np
import pandas as pd
from PIL import Image

Gestures = ['Gesture_0','Gesture_1','Gesture_2','Gesture_3','Gesture_4','Gesture_5',
            'Gesture_6','Gesture_7','Gesture_8','Gesture_9','Gesture_10','Gesture_11']

df = pd.read_csv("/home/shefali/Documents/Declaration_of_consent/Data_recording_codes.csv")
psuedo_codes = list(df['Anonymous code']) #list of codes assigned to the volunteers while collecting data

source_path = "/home/shefali/recorded_data_BB_reduced_size"
destination_path = "/home/shefali/recorded_data_BB_reduced_size_occlusions"

def add_occlusions(image, num_occlusions=1, min_size=5, max_size=15):
    for _ in range(num_occlusions):
        x = np.random.randint(0, image.shape[1]-15) 
        y = np.random.randint(0, image.shape[0]-15)
        size = np.random.randint(min_size, max_size)
        occlusion = np.zeros((size, size, 4), dtype=np.uint8)  # Black occlusion
        image[y:y+size, x:x+size,:] = occlusion
    return image

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
            for key in list(source_file.keys()):
                print(key)
                dataset = source_file[key]
                dataset = dataset[:,:,0:4] 
                updated_dataset = destination_file[key]
                updated_dataset = updated_dataset[:,:,0:4]
                occluded_image = add_occlusions(dataset)
                # Update the dataset with occluded pixel values
                updated_dataset = occluded_image
                print(updated_dataset.shape)
            print(destination_file,'updated')
            #close the .h5py file
            destination_file.close()
            source_file.close()
        print(f'The files for gesture {gesture} is updated')
    print(f'Files for code {code} updated')
print('ALL FILES WITH random occlusion')
            
