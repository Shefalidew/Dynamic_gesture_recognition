import h5py
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

#Normalization step
def scaling(data):
    scaler = MinMaxScaler(feature_range=(0,1))
    print('Y_test',data.shape)
    original_shape = data.shape  # Save the original shape
    reshaped_data = data.reshape(-1, data.shape[-1])  # Reshape to 2D
    # Apply MinMaxScaler
    scaled_data = scaler.fit_transform(reshaped_data.astype(np.float16))
    scaled_data = scaled_data.reshape(original_shape)
    print('Y_test_scaled:',scaled_data.shape)
    np.save('saved_Y_test_scaled.npy',scaled_data)
    return scaled_data

## working code to reduce resolution of all images(all channels) to target_size
s_path = '/home/shefali/recorded_data_BB/'
d_path = '/home/shefali/recorded_data_BB_reduced_size/'
target_size = (96,96)

Gestures = ['Gesture_0','Gesture_1','Gesture_2','Gesture_3','Gesture_4','Gesture_5',
            'Gesture_6','Gesture_7','Gesture_8','Gesture_9','Gesture_10','Gesture_11']

df = pd.read_csv("/home/shefali/Documents/Declaration_of_consent/Data_recording_codes.csv")
psuedo_codes = list(df['Anonymous code']) #list of codes assigned to the volunteers while collecting data
print(psuedo_codes)
code = input('Enter one of the psuedo code')
source_path = os.path.join(s_path,code)
destination_path = os.path.join(d_path,code)
print(code)

for i in Gestures:
    print(i)
    source_path_gesture = os.path.join(source_path,i)
    destination_path_gesture = os.path.join(destination_path,i)
    filenames = os.listdir(source_path_gesture) ##as filenames are same
    for j in filenames:
        source_file_path = os.path.join(source_path_gesture,j)
        destination_file_path = os.path.join(destination_path_gesture,j)
        with h5py.File(source_file_path, 'r') as input_hdf5:
            # Create an HDF5 file for writing
            print('input file:',input_hdf5)
            with h5py.File(destination_file_path, 'a') as output_hdf5:
                # Iterate over the dataset in the input HDF5 file
                for key in input_hdf5.keys():
                    # Read the image from the input HDF5 file
                    print(key)
                    image = input_hdf5[key][:]
                    print(type(image))
                    print(image.shape)

                    # Resize the image to 120x120
                    resized_image = cv2.resize(image, (96,96))
                    # Create a dataset in the output HDF5 file and write the resized image
                    output_hdf5.create_dataset(key, data=resized_image)
                print(output_hdf5,'updated')
        #close the .h5py file
        output_hdf5.close()
        input_hdf5.close()
    print(f'The files for gesture {i} is updated')
