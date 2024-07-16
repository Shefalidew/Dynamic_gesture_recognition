import xgboost as xgb
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
import scipy.stats as stats
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession



if __name__ =='__main__':
    # GPU Initialization
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
    
    param_grid = {
        'max_depth': [3, 5, 7],
        'learning_rate': [0.1, 0.01, 0.001],
        'subsample': [0.5, 0.7, 1],
        'n_estimators' : [25,50,100,150]
    }
    param_dist = {
        'max_depth': stats.randint(3, 20),
        'learning_rate': stats.uniform(0.01, 0.1),
        'subsample': stats.uniform(0.5, 0.5),
        'n_estimators':stats.randint(100, 300)
    }

    xgb_model = xgb.XGBClassifier()
    # Create the RandomizedSearchCV object
    random_search = RandomizedSearchCV(xgb_model, param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy')

    # Fit the RandomizedSearchCV object to the training data
    random_search.fit(X_train, Y_train)

    # Print the best set of hyperparameters and the corresponding score
    print("Best set of hyperparameters: ", random_search.best_params_)
    print("Best score: ", random_search.best_score_)
