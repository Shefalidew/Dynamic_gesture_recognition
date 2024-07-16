import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix 
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



if __name__ =='__main__':
    print('Random Forest training trial 1 with landmarks as dataset')
    landmark_X_train = np.load('Master_Thesis_codes/selected_landmark_X_train.npy',allow_pickle=True)
    print((landmark_X_train.shape))
    # Step 1: Reshape the tensor
    reshaped_tensor = tf.reshape(landmark_X_train, [tf.shape(landmark_X_train)[0], -1])
    print(reshaped_tensor.shape)
    # Step 2: Create a DataFrame (Optional)
    df = pd.DataFrame(reshaped_tensor)
    X_train = df
    print(X_train.shape)
    
    Y_train = np.load('Master_Thesis_codes/extended_translated_Y_train.npy')
    print(Y_train.shape)
    
    landmark_X_test = np.load('Master_Thesis_codes/selected_landmark_X_test.npy',allow_pickle=True)
    print((landmark_X_test.shape))
    # Step 1: Reshape the tensor
    reshaped_tensor = tf.reshape(landmark_X_test, [tf.shape(landmark_X_test)[0], -1])
    print(reshaped_tensor.shape)
    # Step 2: Create a DataFrame (Optional)
    df = pd.DataFrame(reshaped_tensor)
    X_test = df
    print(X_test.shape)
    
    Y_test = np.load('Master_Thesis_codes/extended_translated_Y_test.npy')
    print(Y_test.shape)
    
    # GPU Initialization
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices)) 
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    rf_classifier = RandomForestClassifier(n_estimators=1214,max_depth = None,min_samples_leaf=1, min_samples_split =2, random_state=42,criterion ='entropy',verbose = 1)
    history = rf_classifier.fit(X_train,Y_train)
        
    y_pred = rf_classifier.predict(X_test)
    y_pred =np.argmax(y_pred, axis=1)
    y_test=np.argmax(Y_test, axis=1)
    #Evaluate the model on the validation data for this fold
    categorical_accuracy = accuracy_score(y_test,y_pred)
    print(f" Accuracy = {categorical_accuracy}")
    report = classification_report(y_test, y_pred)
    print(f"Classification Report:\n{report}")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)  

    #training stage finishes here
    
