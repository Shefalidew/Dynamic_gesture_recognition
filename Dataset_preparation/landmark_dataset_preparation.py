## Working code for creating landmark dataset by taking the created RGB dataset as input and replacing landmark.z with depth value

import cv2
import numpy as np
import mediapipe as mp 
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

# Mediapipe starting
#Building landmarks for Holistic approach
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image,model):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) # COLOR CONVERSION BGR 2 RGB
    results = model.process(image) # Make prediction
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR) # COLOR CONVERSION RGB 2 BGR
    return image, results


##Save just the landmark points to be used for training the Mediapipe model

if __name__=='__main__':
    physical_devices = tf.config.list_physical_devices('GPU')
    print("Num GPUs:", len(physical_devices)) 
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)

    X_train = np.load('/home/shefali/Master_Thesis_codes/array_files/extended/extended_BB_X_train.npy')
    RGB_X_train = X_train[:,:,:,:,0:3]
    depth_X_train = X_train[:,:,:,:,-1]

    print('Data loading finished')
    with mp_holistic.Holistic(static_image_mode=False,min_detection_confidence = 0.5, min_tracking_confidence=0.5) as holistic:
        image_data = []
        k=0
        l = 0
        placeholder = np.load('placeholder.npy')
        for i in range(len(RGB_X_train)):
            data = []
            print('i:',i)
            color_image = RGB_X_train[i]
            depth_image = depth_X_train[i]
            for j in range(len(color_image)):
                print('j:',j)
                color_frame = color_image[j]
                depth_frame = depth_image[j]
                image , results = mediapipe_detection(color_frame,holistic)
                Row = []
                if results.pose_landmarks != None:
                    selected_pose_landmarks = results.pose_landmarks.landmark[11:25]
                    for landmark in selected_pose_landmarks:
                        x_landmark = landmark.x*color_frame.shape[0]  # Convert x to image coordinates
                        if x_landmark > 96:
                            x_landmark = 95
                        elif x_landmark < 0:
                            x_landmark = 0
                        y_landmark = landmark.y*color_frame.shape[1] # Convert y to image coordinates
                        if y_landmark > 96:
                            y_landmark = 95
                        elif y_landmark < 0:
                            y_landmark = 0    
                        z_landmark = depth_frame[int(x_landmark), int(y_landmark)]*0.001
                        Row.append([landmark.x, landmark.y, z_landmark]) 
                        placeholder = Row
                        
                else:
                    k = k+1
                    print('K:',k)
                    Row = (placeholder)
                Row = np.array(Row)
                print('Row_shape:',Row.shape)
                data.append(Row) ## for each frame, appended to data
            data = np.array(data)
            print('data_shape:',(data.shape))
            image_data.append(data) ##for each gesture (30 frames), appended to image_data
        landmark_point_array = np.array(image_data)
        print(landmark_point_array.shape)    
        
    cv2.destroyAllWindows()
    np.save('new_replaced_BB_landmark_X_train.npy',landmark_point_array )
    print((landmark_point_array.shape))
    
