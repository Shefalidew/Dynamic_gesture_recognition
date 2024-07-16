import cv2
import numpy as np
import mediapipe as mp 
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

class FullBodyPoseEmbedder(object):
    """Converts 3D pose landmarks into 3D embedding."""

    def __init__(self, torso_size_multiplier=2.5):
        # Multiplier to apply to the torso to get minimal body size.
        self._torso_size_multiplier = torso_size_multiplier

    # Names of the landmarks as they appear in the prediction.
        self._landmark_names = [
            'nose',
            'left_eye_inner', 'left_eye', 'left_eye_outer',
            'right_eye_inner', 'right_eye', 'right_eye_outer',
            'left_ear', 'right_ear',
            'mouth_left', 'mouth_right',
            'left_shoulder', 'right_shoulder',
            'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist',
            'left_pinky_1', 'right_pinky_1',
            'left_index_1', 'right_index_1',
            'left_thumb_2', 'right_thumb_2',
            'left_hip', 'right_hip',
            'left_knee', 'right_knee',
            'left_ankle', 'right_ankle',
            'left_heel', 'right_heel',
            'left_foot_index', 'right_foot_index',
        ]

    def __call__(self, landmarks):
        """Normalizes pose landmarks and converts to embedding

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances defined in `_get_pose_distance_embedding`.
        """
        assert landmarks.shape[0] == len(self._landmark_names), 'Unexpected number of landmarks: {}'.format(landmarks.shape[0])

        # Get pose landmarks.
        landmarks = np.copy(landmarks)

        # Normalize landmarks.
        landmarks = self._normalize_pose_landmarks(landmarks)

        # Get embedding.
        embedding = self._get_pose_distance_embedding(landmarks)

        return embedding

    def _normalize_pose_landmarks(self, landmarks):
        """Normalizes landmarks translation and scale."""
        landmarks = np.copy(landmarks)

        # Normalize translation.
        pose_center = self._get_pose_center(landmarks)
        landmarks -= pose_center

        # Normalize scale.
        pose_size = self._get_pose_size(landmarks, self._torso_size_multiplier)
        landmarks /= pose_size
        # Multiplication by 100 is not required, but makes it eaasier to debug.
        landmarks *= 100

        return landmarks

    def _get_pose_center(self, landmarks):
        """Calculates pose center as point between hips."""
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        center = (left_hip + right_hip) * 0.5
        return center

    def _get_pose_size(self, landmarks, torso_size_multiplier):
        """Calculates pose size.

        It is the maximum of two values:
          * Torso size multiplied by `torso_size_multiplier`
          * Maximum distance from pose center to any pose landmark
        """
        # This approach uses only 2D landmarks to compute pose size.
        landmarks = landmarks[:, :2]

        # Hips center.
        left_hip = landmarks[self._landmark_names.index('left_hip')]
        right_hip = landmarks[self._landmark_names.index('right_hip')]
        hips = (left_hip + right_hip) * 0.5

        # Shoulders center.
        left_shoulder = landmarks[self._landmark_names.index('left_shoulder')]
        right_shoulder = landmarks[self._landmark_names.index('right_shoulder')]
        shoulders = (left_shoulder + right_shoulder) * 0.5

        # Torso size as the minimum body size.
        torso_size = np.linalg.norm(shoulders - hips)

        # Max dist to pose center.
        pose_center = self._get_pose_center(landmarks)
        max_dist = np.max(np.linalg.norm(landmarks - pose_center, axis=1))

        return max(torso_size * torso_size_multiplier, max_dist)

    def _get_pose_distance_embedding(self, landmarks):
        """Converts pose landmarks into 3D embedding.

        We use several pairwise 3D distances to form pose embedding. All distances
        include X and Y components with sign. We differnt types of pairs to cover
        different pose classes. Feel free to remove some or add new.

        Args:
          landmarks - NumPy array with 3D landmarks of shape (N, 3).

        Result:
          Numpy array with pose embedding of shape (M, 3) where `M` is the number of
          pairwise distances.
        """
        embedding = np.array([
            # One joint.
            self._get_distance(
                self._get_average_by_names(landmarks, 'left_hip', 'right_hip'),
                self._get_average_by_names(landmarks, 'left_shoulder', 'right_shoulder')),

            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_elbow'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_elbow'),

            self._get_distance_by_names(landmarks, 'left_elbow', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_elbow', 'right_wrist'),
            
            self._get_distance_by_names(landmarks, 'left_wrist','left_thumb_2' ),
            self._get_distance_by_names(landmarks, 'right_wrist','right_thumb_2'),
            
            self._get_distance_by_names(landmarks, 'left_wrist','left_pinky_1' ),
            self._get_distance_by_names(landmarks, 'right_wrist','right_pinky_1'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_knee'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_knee'),

            # Two joints.
            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_wrist'),
            
            #Three joints.
            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_pinky_1'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_pinky_1'),
            
            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_index_1'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_index_1'),
            
            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_thumb_2'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_thumb_2'),
            
            self._get_distance_by_names(landmarks, 'left_hip', 'left_elbow'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_elbow'),

            # Four joints.
            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),
            
            self._get_distance_by_names(landmarks, 'left_hip', 'left_index_1'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_index_1'),
            
            self._get_distance_by_names(landmarks, 'left_hip', 'left_pinky_1'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_pinky_1'),
            
            self._get_distance_by_names(landmarks, 'left_hip', 'left_thumb_2'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_thumb_2'),

            # Five joints.
            self._get_distance_by_names(landmarks, 'left_shoulder', 'left_ankle'),
            self._get_distance_by_names(landmarks, 'right_shoulder', 'right_ankle'),

            self._get_distance_by_names(landmarks, 'left_hip', 'left_wrist'),
            self._get_distance_by_names(landmarks, 'right_hip', 'right_wrist'),

            # Cross body.

            self._get_distance_by_names(landmarks, 'left_elbow', 'right_elbow'),
            self._get_distance_by_names(landmarks, 'left_knee', 'right_knee'),

            self._get_distance_by_names(landmarks, 'left_wrist', 'right_wrist'),
            self._get_distance_by_names(landmarks, 'left_ankle', 'right_ankle'),
        
            # Body bent direction.

            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'left_wrist', 'left_ankle'),
            #     landmarks[self._landmark_names.index('left_hip')]),
            # self._get_distance(
            #     self._get_average_by_names(landmarks, 'right_wrist', 'right_ankle'),
            #     landmarks[self._landmark_names.index('right_hip')]),
        ])

        return embedding

    def _get_average_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return (lmk_from + lmk_to) * 0.5

    def _get_distance_by_names(self, landmarks, name_from, name_to):
        lmk_from = landmarks[self._landmark_names.index(name_from)]
        lmk_to = landmarks[self._landmark_names.index(name_to)]
        return self._get_distance(lmk_from, lmk_to)

    def _get_distance(self, lmk_from, lmk_to):
        return lmk_to - lmk_from
    

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
    
    X_test = np.load('/home/shefali/Master_Thesis_codes/array_files/extended/extended_BB_X_test.npy')
    RGB_X_test = X_test[:,:,:,:,0:3]
    
    print('Data loading finished')
    with mp_holistic.Holistic(min_detection_confidence = 0.5, min_tracking_confidence=0.8) as holistic:
        image_data = []
        k=0
        for i in range(len(RGB_X_test)):
            data = []
            print('i:',i)
            color_image = RGB_X_test[i]
            for j in range(len(color_image)):
                color_frame = color_image[j]
                print(color_frame.shape)
                image , results = mediapipe_detection(color_frame,holistic)
                Row = []
                # Convert the BGR image to RGB before processing.
                #results = holistic.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
                if results.pose_world_landmarks != None:
                    selected_pose_landmarks = results.pose_world_landmarks.landmark[11:25]
                    for landmark in selected_pose_landmarks:
                        Row.append([landmark.x, landmark.y, landmark.z]) 
                        ##(x,y,z) for each landmark appended to Row
                        placeholder = Row
                else:
                    k = k+1
                    print('K:',k)
                    '''There are some frames where 
                    Mediapipe is unable to create landmarks, 
                    in that case fill the values with landmarks from previous frame'''
                    Row = (placeholder)
                Row = np.array(Row)
                print('Row_shape:',Row.shape)
                fb = FullBodyPoseEmbedder()
                embedding = fb.__call__(Row)
                print('Embedding_shape:',embedding.shape)
                data.append(embedding) ## for each frame, appended to data
            data = np.array(data)
            print('data_shape:',(data.shape))
            image_data.append(data) ##for each gesture (30 frames), appended to image_data
        landmark_point_array = np.array(image_data)
        print(landmark_point_array.shape)    
                
    cv2.destroyAllWindows()
