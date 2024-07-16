## code to be used to record my gestures as separate files named by gestures

import pyrealsense2 as rs
import numpy as np
import cv2
import os
import h5py
import csv 
from datetime import datetime


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


def main(name=None,code=None):
    frames_capture = 30
    frame_number = 0
    if name==None:
        name = input('Enter the Name of Gesture: ')
    if code==None:
        code = input('Enter the psudeo code: ')
    path = 'recorded_data/'+name
    if not os.path.exists(path):
        os.makedirs(path)

    #saving data as .csv file and .h5 file
    dirname = path+'/'+datetime.now().strftime("%Y_%m_%d-%H_%M_%S")
    csvfile = dirname+'.csv'
    f =  open(csvfile, 'w', encoding='UTF8') 
    h5f = h5py.File(dirname+'.h5', 'w') 
    writer = csv.writer(f, delimiter=';')
    csvHeader = np.array(['Frame', 'RGB Data', 'Depth Data'])
    writer.writerow(csvHeader)
    
    
    while frame_number<=frames_capture:
        # Wait for a coherent pair of frames: depth and color
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())
            
        writer.writerow([frame_number, color_image.tolist(), depth_image.tolist()])
        img_data = np.dstack((color_image, depth_image)).astype(np.uint8)
        h5f.create_dataset('frame_'+str(frame_number), data=img_data)
        frame_number += 1
            
        #show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.putText(img=color_image, text=name, org=(10, 30), fontFace=cv2.FONT_HERSHEY_TRIPLEX, fontScale=1, color=(0, 255, 0),thickness=1)
        cv2.imshow('RealSense', color_image)

        # Break gracefully
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
                
    f.close()
    h5f.close()
    cv2.destroyAllWindows()
        
        
        
if __name__ == '__main__':
    code = input('Enter the psuedo code: ')
    input('Starting Gesture_0...')
    for i in range(10):
        main(name='Gesture_0',code=code)
        print(f'Recorded {i}th set for: Gesture_0')
        input('CONTINUE')
    input('Starting Gesture_1...')
    for i in range(10):
        main(name='Gesture_1',code=code)
        print(f'Recorded {i}th set for for: Gesture_1')
        input('CONTINUE')
    input('Starting Gesture_2...')
    for i in range(10):
        main(name='Gesture_2',code=code)
        print(f'Recorded {i}th set for for: Gesture_2')
        input('CONTINUE')
    input('Starting Gesture_3...')
    for i in range(10):
        main(name='Gesture_3',code=code)
        print(f'Recorded {i}th set for for: Gesture_3')
        input('CONTINUE')
    input('Starting Gesture_4...')
    for i in range(10):
        main(name='Gesture_4',code=code)
        print(f'Recorded {i}th set for for: Gesture_4')
        input('CONTINUE')
    input('Starting Gesture_5...')
    for i in range(10):
        main(name='Gesture_5',code=code)
        print(f'Recorded {i}th set for Gesture_5')
        input('CONTINUE')
    input('Starting Gesture_6...')
    for i in range(10):
        main(name='Gesture_6',code=code)
        print(f'Recorded {i}th set for for: Gesture_6')
        input('CONTINUE')
    input('Starting Gesture_7...')
    for i in range(10):
        main(name='Gesture_7',code=code)
        print(f'Recorded {i}th set for for: Gesture_7')
        input('CONTINUE')
    input('Starting Gesture_8...')
    for i in range(10):
        main(name='Gesture_8',code=code)
        print(f'Recorded {i}th set for Gesture_8')
        input('CONTINUE')
    input('Starting Gesture_9...')
    for i in range(10):
        main(name='Gesture_9',code=code)
        print(f'Recorded {i}th set for for: Gesture_9')
        input('CONTINUE')
    input('Starting Gesture_10...')
    for i in range(10):
        main(name='Gesture_10',code=code)
        print(f'Recorded {i}th set for Gesture_10')
        input('CONTINUE')
    input('Starting Gesture_11...')
    for i in range(10):
        main(name='Gesture_11',code=code)
        print(f'Recorded {i}th set for Gesture_11')
        input('CONTINUE')
    # Stop streaming
    pipeline.stop()
        
    
