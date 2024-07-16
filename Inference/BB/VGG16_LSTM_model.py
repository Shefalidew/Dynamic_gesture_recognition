## working code of Inference for VGG16+LSTM model on BB clipped images

import cv2
import time
import torch
import warnings
import numpy as np
import pyrealsense2 as rs
from pickle import TRUE
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.preprocessing import MinMaxScaler

Labels = ['Gesture_0','Gesture_1','Gesture_2','Gesture_3','Gesture_4','Gesture_5',
            'Gesture_6','Gesture_7','Gesture_8','Gesture_9','Gesture_10','Gesture_11']

# Initialize Realsense
# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
if device_product_line == 'L500':
    config.enable_stream(rs.stream.color, 960, 540, rs.format.bgr8, 30)
else:
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

# GPU Initialization
physical_devices = tf.config.list_physical_devices('GPU') 
print("Num GPUs:", len(physical_devices)) 
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.9
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

model = models.load_model('/home/shefali/Master_Thesis_codes/Working_trained_models/Inference/VGG16_BB_rgbmodel_TL.h5',compile=False)
# Load YOLOv5 model
yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
        

class ExtractBoudingBoxRegion():
    def __init__(self,image):
        self.image = image
         
    def detect_with_yolov5(self):
        # Perform object detection
        self.results = yolo_model(self.image)
        return self.results

    # Detect persons using different pre-trained models
    def detect_persons(self):
        self.yolov5_results = self.detect_with_yolov5()
        self.yolov5_bboxes = self.yolov5_results.xyxy[0].cpu().numpy()
        self.labels = self.yolov5_results.names
        return self.yolov5_bboxes,self.labels

    # Display bounding boxes around detected persons
    def get_bounding_boxes(self, boxes, labels):
        self.label_to_show = 0  #Label person has key 0
        for bbox, label in zip(boxes,labels):
            # Check if the label matches the one to display
            if label == self.label_to_show:
                self.xmin, self.ymin, self.xmax, self.ymax = bbox[:4] #person found
        return self.xmin, self.ymin, self.xmax, self.ymax

    def get_cropped_image(self):
        self.boxes ,self.labels = self.detect_persons()
        self.Xmin,self.Ymin,self.Xmax,self.Ymax = self.get_bounding_boxes(self.boxes,self.labels)
        if (self.Xmin < 20 or self.Ymin < 20 or self.Xmax >620 or self.Ymax > 460):
            self.cropped_image_values = self.image[int(self.Ymin):int(self.Ymax),int(self.Xmin):int(self.Xmax),:]
        else:
            self.cropped_image_values = self.image[int(self.Ymin)-20:int(self.Ymax)+20,int(self.Xmin)-20:int(self.Xmax)+20,:]
        return self.cropped_image_values

def reduce_size(image):
    image_reduced = cv2.resize(cropped_image, (96,96))
    return image_reduced
    
def scaling(img_data):
    scaler = MinMaxScaler(feature_range=(0,1))
    #scaling required [0,1]
    original_shape = img_data.shape  # Save the original shape
    reshaped_data = img_data.reshape(-1, img_data.shape[-1])  # Reshape to 2D
    scaled_data = scaler.fit_transform(reshaped_data.astype(np.float16))
    img_data = scaled_data.reshape(original_shape)
    return img_data

if __name__ == '__main__':
    frames_capture = 31
    cap = cv2.VideoCapture(5)
    pred = ''
    while TRUE:
        i = 0
        data = []
        start_time = time.time()  # Record the start time    
        while i < frames_capture:
            #Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame() 
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            color_image = np.array(color_frame.get_data())    
            eBB =ExtractBoudingBoxRegion(color_image)
            cropped_image = eBB.get_cropped_image()
            image_reduced = reduce_size(cropped_image)
            data.append(image_reduced)
            i=i+1
            
            image = cv2.putText(img=color_image, text='iter:'+str(i)+'    Prediction: '+pred, org=(50, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=1)

            # Show to screen
            cv2.imshow('OpenCV Feed', color_image)

            # Break gracefully
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break
            
        img_data = np.array(data) 
        img_data = scaling(img_data)
        img_data = img_data.reshape((1,img_data.shape[0], img_data.shape[1],img_data.shape[2],img_data.shape[3]))
        
        predict = model.predict(img_data)
        end_time = time.time()  # Record the end time
        inference_time = end_time - start_time  # Calculate the time difference
        print(f'{inference_time}')

        y_pred_integer = np.argmax(predict, axis=1)

        if predict[0][y_pred_integer[0]]>0.6:
            pred = Labels[y_pred_integer[0]]
        else:
            pred = ''
    
    cap.release()
    cv2.destroyAllWindows()
