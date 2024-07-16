##working code for Inference on MediapipeLSTM+XGBoost classifier model using selected landmarks(replacedZ)
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


Labels = ['Gesture_0','Gesture_1','Gesture_2','Gesture_3','Gesture_4','Gesture_5',
            'Gesture_6','Gesture_7','Gesture_8','Gesture_9','Gesture_10','Gesture_11']

mediapipelstm = load_model('/home/shefali/Master_Thesis_codes/Working_trained_models/Inference/replacedZ_mediapipe_model.h5')
loaded_model = xgb.Booster(model_file='/home/shefali/Master_Thesis_codes/Working_trained_models/XGBoost_model_with_mediapipeLSTM.json')
intermediate_layer_model = Model(inputs=mediapipelstm.get_layer('bidirectional').input,outputs=mediapipelstm.get_layer('bidirectional_2').output)

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

def mediapipe_detection(image, model):
	image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
	#image.flags.writable = False				 # Image is no longer writable
	results = model.process(image)				 # Make prediction
	#image.flags.writable = True				 # Image is now writable
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
def extract_features(img_data):
    train_features = intermediate_layer_model.predict(img_data)
    df = pd.DataFrame(train_features)
    return df

if __name__ == '__main__':
    frames_capture = 31
    cap = cv2.VideoCapture(5)
    with mp_holistic.Holistic(min_detection_confidence=0.9, min_tracking_confidence=0.9) as holistic:
        pred = ''
        while TRUE:
            
            i = 0
            data = []
            img_data = []
            start_time = time.time()  # Record the start time
            while i < frames_capture:
                # Wait for a coherent pair of frames: depth and color
                frames = pipeline.wait_for_frames()
                depth_frame = frames.get_depth_frame()
                color_frame = frames.get_color_frame()
                if not depth_frame or not color_frame:
                    continue
                
                # Convert images to numpy arrays
                depth_image = np.asanyarray(depth_frame.get_data())
                color_image = np.asanyarray(color_frame.get_data())
                
                # Make detections
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
                        z_landmark = depth_image[int(x_landmark), int(y_landmark)]*0.001  # Get depth in meters
                        Row.append([landmark.x, landmark.y, z_landmark]) 
                    Row = np.array(Row)
                    data.append(Row) ## for each frame, appended to data
                    i = i+1
                # Draw landmarks
                draw_styled_landmarks(image, results)

                image = cv2.putText(img=image, text='iter:'+str(i)+f'    Prediction: {pred}', org=(50, 50), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=1)

                # Show to screen
                cv2.imshow('OpenCV Feed', image)

                # Break gracefully
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break
            
            img_data = np.array(data)
            img_data = img_data.reshape((1,img_data.shape[0], img_data.shape[1]*img_data.shape[2]))
            df = extract_features(img_data)
            predict = loaded_model.predict(xgb.DMatrix(df)) #For XGBoostClassifiers
            pred = Labels[int(predict[0])]
            end_time = time.time()  # Record the end time
            inference_time = end_time - start_time  # Calculate the time difference
            print(inference_time)
    
        
    
    cap.release()
    cv2.destroyAllWindows()
