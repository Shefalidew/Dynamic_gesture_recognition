##Inference of Stacking (MLP) model using models trained on BB clipped images
import cv2
import time
import torch
import warnings
import numpy as np
import pandas as pd
import mediapipe as mp
import xgboost as xgb
import pyrealsense2 as rs
from pickle import TRUE
import tensorflow as tf
from keras.models import Model, load_model
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

mp_holistic = mp.solutions.holistic # Holistic model
mp_drawing = mp.solutions.drawing_utils # Drawing utilities

cnnlstm = models.load_model('/home/shefali/Master_Thesis_codes/Working_trained_models/Inference/CNNLSTM_BB_trained.h5',compile=False)
mediapipelstm = models.load_model('/home/shefali/Master_Thesis_codes/Working_trained_models/Inference/selected_BB_mediapipeModel4.h5',compile=False)
cnnlstmXGB = xgb.Booster(model_file='/home/shefali/Master_Thesis_codes/Working_trained_models/Inference/XGBoost_model_with_CNNLSTM2.json')
intermediate_layer_model = Model(inputs=cnnlstm.get_layer('input_1').input,outputs=cnnlstm.get_layer('bidirectional_2').output)

base_models = [cnnlstm, mediapipelstm,cnnlstmXGB]

MLP_model = models.load_model('/home/shefali/Master_Thesis_codes/Working_trained_models/Meta_MLP_model_BB.h5',compile=False)
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
        # YOLOv5
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

def images_from_frame(color_frame,depth_frame):
    # Convert images to numpy arrays
    depth_image = np.array(depth_frame.get_data())
    depth_image = depth_image.astype(np.uint8) ## As our depth values were stored in uint8 format
    depth_image = np.expand_dims(depth_image, axis=-1)
    color_image = np.array(color_frame.get_data())
    combined_data = np.concatenate((color_image,depth_image), axis=-1)
    return color_image,depth_image,combined_data

def reduce_size(image):
    image_reduced = cv2.resize(cropped_image, (96,96))
    return image_reduced
    
def extract_features(img_data):
    train_features = intermediate_layer_model.predict(img_data)
    df = pd.DataFrame(train_features)
    return df

def scaling(img_data):
    scaler = MinMaxScaler(feature_range=(0,1))
    #scaling required [0,1]
    original_shape = img_data.shape  # Save the original shape
    reshaped_data = img_data.reshape(-1, img_data.shape[-1])  # Reshape to 2D
    scaled_data = scaler.fit_transform(reshaped_data.astype(np.float16))
    img_data = scaled_data.reshape(original_shape)
    return img_data

def meta_learning(base_models,cnnlstm_data, mediapipelstm_data):
    meta_train = np.zeros((1, len(base_models)))
    pred = base_models[0].predict(cnnlstm_data)
    pred = np.argmax(pred,axis=1)
    meta_train[:,0] = pred #Prediction value of cnnLSTM
    pred = base_models[1].predict(mediapipelstm_data)
    pred = np.argmax(pred,axis=1)
    meta_train[:,1] = pred #Prediction value of mediapipeLSTM
    df = extract_features(cnnlstm_data)
    meta_train[:,2] = base_models[2].predict(xgb.DMatrix(df)) #Prediction value of mediapipeXGB
    return meta_train

def mediapipe_detection(image, model):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
	results = model.process(image)				 # Make prediction
	image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
	return image, results

def draw_landmarks(image, results):
	mp_drawing.draw_landmarks(
	image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS) # Draw pose connections
	
def draw_styled_landmarks(image, results):
	# Draw pose connections
	mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
							mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
							mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
							)


if __name__ == '__main__':
    frames_capture = 31
    cap = cv2.VideoCapture(5)
    
    with mp_holistic.Holistic(min_detection_confidence=0.9, min_tracking_confidence=0.9) as holistic:
        pred = ''
        while TRUE:
            i = 0
            cnnlstm_data = []
            mediapipelstm_data = []
            start_time = time.time()  # Record the start time    
            while i < frames_capture:
    			# Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                    
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                    
                color_image,depth_image,combined_data = images_from_frame(color_frame,depth_frame)
                eBB =ExtractBoudingBoxRegion(combined_data)
                cropped_image = eBB.get_cropped_image()
                image_reduced = reduce_size(cropped_image)
                cnnlstm_data.append(image_reduced) #RGBD input for cnnlstm model
                
                image, results = mediapipe_detection(color_image, holistic)
                Row = []                
                if results.pose_landmarks != None:
                    selected_pose_landmarks = results.pose_landmarks.landmark[11:25]
                    for landmark in selected_pose_landmarks:
                        Row.append([landmark.x, landmark.y, landmark.z])
                    Row = np.array(Row)
                    mediapipelstm_data.append(Row) ## for each frame, appended to data
                i=i+1
                # Draw landmarks
                draw_styled_landmarks(image, results)
                image = cv2.putText(img=image, text='iter:'+str(i)+'    Prediction: '+pred, org=(50, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=1)
                # Show to screen
                cv2.imshow('OpenCV Feed', image)
                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
                
            cnnlstm_img_data = np.array(cnnlstm_data) 
            cnnlstm_img_data = scaling(cnnlstm_img_data)
            cnnlstm_img_data = cnnlstm_img_data.reshape((1,cnnlstm_img_data.shape[0], cnnlstm_img_data.shape[1],cnnlstm_img_data.shape[2],cnnlstm_img_data.shape[3]))
            
            mediapipelstm_data = np.array(mediapipelstm_data) 
            mediapipelstm_data = mediapipelstm_data.reshape((1,mediapipelstm_data.shape[0], mediapipelstm_data.shape[1]*mediapipelstm_data.shape[2]))
           
            try:
                meta_data = meta_learning(base_models,cnnlstm_img_data,mediapipelstm_data)
                end_time = time.time()  # Record the end time
                predict = MLP_model.predict(meta_data)
            except ValueError:
                continue
            
            inference_time = end_time - start_time  # Calculate the time difference
            print(inference_time)
            y_pred_integer = np.argmax(predict, axis=1)
    
            if predict[0][y_pred_integer[0]]>0.6:
                pred = Labels[y_pred_integer[0]]
            else:
                pred = ''
                
    cap.release()
    cv2.destroyAllWindows()
