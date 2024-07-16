import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix 
from sklearn.preprocessing import LabelBinarizer
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



if __name__ =='__main__':
    # GPU Initialization
    print('XGBoost for the landmarks obtained from the BB clipped images')
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices)) 
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    landmark_X_train = np.load('Master_Thesis_codes/selected_landmark_BB_X_train.npy',allow_pickle=True)
    print((landmark_X_train.shape))
    # Step 1: Reshape the tensor
    reshaped_tensor = tf.reshape(landmark_X_train, [tf.shape(landmark_X_train)[0], -1])
    print(reshaped_tensor.shape)
    # Step 2: Create a DataFrame (Optional)
    df = pd.DataFrame(reshaped_tensor)
    X_train = df
    print(X_train.shape)
    
    Y_train = np.load('Master_Thesis_codes/extended_BB_Y_train.npy')
    Y_train = np.argmax(Y_train,axis=1)
    print(Y_train.shape)
    
    landmark_X_test = np.load('Master_Thesis_codes/selected_landmark_BB_X_test.npy',allow_pickle=True)
    print((landmark_X_test.shape))
    # Step 1: Reshape the tensor
    reshaped_tensor = tf.reshape(landmark_X_test, [tf.shape(landmark_X_test)[0], -1])
    print(reshaped_tensor.shape)
    # Step 2: Create a DataFrame (Optional)
    df = pd.DataFrame(reshaped_tensor)
    X_test = df
    print(X_test.shape)
    
    Y_test = np.load('Master_Thesis_codes/extended_BB_Y_test.npy')
    Y_test = np.argmax(Y_test,axis=1)
    print(Y_test.shape)
    
    #hyperparameters found using the one of the search method
    clf = xgb.XGBClassifier(num_class=12,tree_method='hist',learning_rate= 0.05656701659708586, max_depth= 7, n_estimators= 227, subsample= 0.8005696737667015)
    history = clf.fit(X_train,Y_train)
    print(clf)
    clf.save_model('XGBoost_BB_landmark_model.json')
    #clf = load('HistGB_classifier.joblib')
    score = clf.score(X_test,Y_test)
    print(score)  
    y_pred = clf.predict(X_test)

    #Evaluate the model on the validation data for this fold
    categorical_accuracy = accuracy_score(Y_test,y_pred)
    print(f" Accuracy = {categorical_accuracy}")
    report = classification_report(Y_test, y_pred)
    print(f"Classification Report:\n{report}")
    cm = confusion_matrix(Y_test, y_pred)
    print(cm)  
    
    
