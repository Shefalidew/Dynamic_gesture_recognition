## working code to clip the image up to BB only for all the files in the folders

import h5py
import os
import cv2
from PIL import Image
import numpy as np
import pandas as pd
import torch
from torchvision import transforms

class ExtractBoudingBoxRegion():
    def __init__(self,image):
        self.image = image
        
        
    def detect_with_yolov5(self,image):
        # Load YOLOv5 model
        self.model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        # Perform object detection
        self.results = self.model(image)
        return self.results

    # Detect persons using different pre-trained models
    def detect_persons(self,image):
        # YOLOv5
        self.yolov5_results = self.detect_with_yolov5(image)
        print(self.yolov5_results)
        self.yolov5_bboxes = self.yolov5_results.xyxy[0].cpu().numpy()
        self.labels = self.yolov5_results.names
        return self.yolov5_bboxes,self.labels

    # Display bounding boxes around detected persons
    def get_bounding_boxes(self,image, boxes, labels):
        self.label_to_show = 0  #Label person has key 0
        for bbox, label in zip(boxes,labels):
            # Check if the label matches the one to display
            if label == self.label_to_show:
                print('person found')
                self.xmin, self.ymin, self.xmax, self.ymax = bbox[:4]
           
        return self.xmin, self.ymin, self.xmax, self.ymax

if __name__=='__main__':
    s_path = '/home/shefali/recorded_data_batch/'
    d_path = '/home/shefali/recorded_data_BB/'
        
    Gestures = ['Gesture_0','Gesture_1','Gesture_2','Gesture_3','Gesture_4','Gesture_5',
            'Gesture_6','Gesture_7','Gesture_8','Gesture_9','Gesture_10','Gesture_11']

    df = pd.read_csv("/home/shefali/Documents/Declaration_of_consent/Data_recording_codes.csv")
    psuedo_codes = list(df['Anonymous code']) #list of codes assigned to the volunteers while collecting data
    print(psuedo_codes)
    code = input('Enter one of the pseudo code')
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
                    with h5py.File(destination_file_path, 'a') as output_hdf5:
                        # Iterate over the dataset in the input HDF5 file
                        for key in input_hdf5.keys():
                            # Read the image from the input HDF5 file
                            image = input_hdf5[key][:]
                            eBB = ExtractBoudingBoxRegion(image)
                            boxes ,labels = eBB.detect_persons(image)
                            Xmin,Ymin,Xmax,Ymax = eBB.get_bounding_boxes(image,boxes,labels)
                            print(Xmin,Ymin,Xmax,Ymax)
                            if (Xmin < 20 or Ymin < 20 or Xmax >620 or Ymax > 460):
                                cropped_image_values = image[int(Ymin):int(Ymax),int(Xmin):int(Xmax),:]
                            else:
                                cropped_image_values = image[int(Ymin)-20:int(Ymax)+20,int(Xmin)-20:int(Xmax)+20,:]
                            print(cropped_image_values.shape)
                            #updated_dataset = np.zeros((cropped_image_values.shape[0],cropped_image_values.shape[1],cropped_image_values.shape[-1]))
                            #updated_dataset[:] = cropped_image_values
                            output_hdf5.create_dataset(key, data=cropped_image_values)
                        print(output_hdf5,'updated')
            #close the .h5py file
            output_hdf5.close()
            input_hdf5.close()
        print(f'The files for gesture {i} is updated')

    
