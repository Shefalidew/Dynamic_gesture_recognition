##Inference of Stacking (XGBoost) model
##working code for Inference on Mediapipe model
import cv2
import time
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb
import mediapipe as mp
from joblib import load
from pickle import TRUE
import pyrealsense2 as rs
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.models import Model, load_model
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

#loading the models
cnnlstm = load_model('/home/shefali/Master_Thesis_codes/Working_trained_models/final_CNN_RNN_model.h5')
mediapipelstm = load_model('/home/shefali/Master_Thesis_codes/Working_trained_models/Inference/replacedZ_mediapipe_model.h5')
mediapipeXGB = xgb.Booster(model_file='/home/shefali/Master_Thesis_codes/Working_trained_models/XGBoost_model_with_mediapipeLSTM.json')
intermediate_layer_model = Model(inputs=mediapipelstm.get_layer('bidirectional').input,outputs=mediapipelstm.get_layer('bidirectional_2').output)
base_models = [cnnlstm, mediapipelstm,mediapipeXGB]
#loading meta learning model
xgb_meta = xgb.Booster(model_file='/home/shefali/Master_Thesis_codes/Working_trained_models/Meta_learner_model.json')


def images_from_frame(color_frame,depth_frame):
    # Convert images to numpy arrays
    depth_image = np.asanyarray(depth_frame.get_data())
    depth_image_reduced = cv2.resize(depth_image, (96,96))
    depth_image_reduced = np.expand_dims(depth_image_reduced, axis=-1)
    color_image = np.asanyarray(color_frame.get_data())
    color_image_reduced = cv2.resize(color_image, (96,96))
    combined_data = np.concatenate((color_image_reduced,depth_image_reduced), axis=-1)
    return color_image,depth_image,combined_data

def extract_features(img_data):
    train_features = intermediate_layer_model.predict(img_data)
    df = pd.DataFrame(train_features)
    return df

def meta_learning(base_models,cnnlstm_data, mediapipelstm_data):
    meta_train = np.zeros((1, len(base_models)))
    pred = base_models[0].predict(cnnlstm_data)
    pred = np.argmax(pred,axis=1)
    meta_train[:,0] = pred #Prediction value of cnnLSTM
    pred = base_models[1].predict(mediapipelstm_data)
    pred = np.argmax(pred,axis=1)
    meta_train[:,1] = pred #Prediction value of mediapipeLSTM
    df = extract_features(mediapipelstm_data)
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
    
    with mp_holistic.Holistic(min_detection_confidence=0.9, min_tracking_confidence=0.9) as holistic:
        pred = ''
        while TRUE:
            i = 0
            cnnlstm_data = []
            mediapipelstm_data = []
            #start_time = time.time()  # Record the start time    
            while i < frames_capture:
    			# Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                    
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                   
                color_image,depth_image,combined_data = images_from_frame(color_frame,depth_frame)
                cnnlstm_data.append(combined_data) #RGBD input for cnnlstm model
                
                image, results = mediapipe_detection(color_image, holistic)
                Row = []                
                if results.pose_landmarks != None:
                    selected_pose_landmarks = results.pose_landmarks.landmark[11:25]
                    for landmark in selected_pose_landmarks:
                        x_landmark = landmark.x*color_image.shape[0]  # Convert x to image coordinates
                        if x_landmark > 96:
                            x_landmark = 95
                        elif x_landmark < 0:
                            x_landmark = 0
                        y_landmark = landmark.y*color_image.shape[1] # Convert y to image coordinates
                        if y_landmark > 96:
                            y_landmark = 95
                        elif y_landmark < 0:
                            y_landmark = 0    
                        z_landmark = depth_image[int(x_landmark), int(y_landmark)]*0.001 # Get depth in meters
                        Row.append([landmark.x, landmark.y, z_landmark])
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
            
            meta_data = meta_learning(base_models,cnnlstm_img_data,mediapipelstm_data)
            
            start_time = time.time()  # Record the start time
            predict = xgb_meta.predict(xgb.DMatrix(meta_data))
            pred = Labels[int(predict[0])]
            end_time = time.time()  # Record the end time
            inference_time = end_time - start_time  # Calculate the time difference
            print(inference_time)
                
    cap.release()
    cv2.destroyAllWindows()
