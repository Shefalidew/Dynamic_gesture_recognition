import numpy as np
import pandas as pd
import tensorflow as tf
from scipy.stats import randint
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
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
    
    #for Grid Search algorithm
    '''param_grid = {'n_estimators': [100, 200, 300,500,700,800,1000],
    'max_depth': [None, 5, 15,35,50,75,100],
    'min_samples_split': [2,3, 5, 10],
    'min_samples_leaf': [1, 2, 4]
    }'''
    ##for Randomized Search algorithm
    param_dist = {'n_estimators': randint(100, 1500),
    'max_depth': [None] + list(randint(1, 100).rvs(10)),
    'min_samples_split': randint(2, 11),
    'min_samples_leaf': randint(1, 11)
    }
    
    rf_classifier = RandomForestClassifier(random_state=42,criterion ='entropy')
    #grid_search = GridSearchCV(estimator=rf_classifier, param_grid=param_grid, cv=5, n_jobs=-1)
    random_search = RandomizedSearchCV(estimator=rf_classifier, param_distributions=param_dist, cv=5, n_iter=100, n_jobs=-1)
    history = random_search.fit(X_train,Y_train)
    print("Best hyperparameters:", random_search.best_params_)
        

    #training stage finishes here
    
