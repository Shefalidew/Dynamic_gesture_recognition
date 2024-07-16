## model with reduced complexity and Dropout layers
import numpy as np
import json
import datetime
import tensorflow as tf
from sklearn.utils import resample
from tensorflow.keras import layers, models
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.models import Model, load_model
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix 



def majority_voting_ensemble(models,X_test):
    predictions = []
    for model in models:
        predictions.append(model.predict(X_test))
        predictions = np.asarray(predictions) 
        ## here take predict_proba not argmax, then you will be having a 2D matrix of size (5,12). 
        ##Calculate avg of each column and retuurn the column with max avg as final prediction.
    print(predictions.shape)
    #aggregated_predictions
    ensemble_preds = np.zeros((len(predictions[0])))
    print(f'ensemble_preds = {ensemble_preds.shape}')
    for i in range(len(predictions[0])):
        votes = np.stack([preds[i] for preds in predictions],axis =0)
        print('votes:',votes)
        ensemble_preds[i] = np.argmax(np.bincount(votes))
        print(ensemble_preds[i])
    print(ensemble_preds.shape)
    return ensemble_preds
    
def soft_voting(models,X_test):
    all_probs = [model.predict(X_test) for model in models]
    print(np.asarray(all_probs).shape)
    avg_probs = np.mean(all_probs,axis=0)
    print(avg_probs)
    print(type(avg_probs))
    predicted_classes = np.argmax(avg_probs,axis=1)
    return predicted_classes

if __name__ =='__main__':
    print('Performing soft voting ensemble on trained CNN+LSTM models')
    # GPU Initialization
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices)) 
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    
    X_train_array = np.load('Master_Thesis_codes/saved_extended_occluded_X_train_scaled.npy')
    print((X_train_array.shape))
    RGB_X_train = X_train_array[:,:,:,:,0:3]
    print(f'RGB_X_train shape: {RGB_X_train.shape}')
    
    
    y_train = np.load('Master_Thesis_codes/saved_extended_occluded_Y_train_scaled.npy')
    print(y_train.shape)
    
    X_test_array = np.load('Master_Thesis_codes/saved_extended_occluded_X_test_scaled.npy')
    print((X_test_array.shape))
    RGB_X_test = X_test_array[:,:,:,:,0:3]
    print(f'RGB_X_test shape: {RGB_X_test.shape}')
    
    y_test = np.load('Master_Thesis_codes/saved_extended_occluded_Y_test_scaled.npy')
    print(y_test.shape)
    print('Data loading finished')
    
    #bagged_models = bagging_ensemble(RGB_X_train,y_train,n_models = 5)
    
    print('Soft Majority voting ensmeble prediction')
    n_models = 5
    num_classes = 12
    model_1 = load_model('kfold_for_cnnLSTM_bagging_4.h5')
    model_2 = load_model('kfold_for_cnnLSTM_bagging1_3.h5')
    model_3 = load_model('kfold_for_cnnLSTM_bagging1_4.h5')
    model_4 = load_model('kfold_for_cnnLSTM_bagging2_0.h5')
    model_5 = load_model('kfold_for_cnnLSTM_bagging2_1.h5')
    bagged_models = [model_1,model_2,model_3,model_4,model_5]
    #bagged_models = [load_model(f'bagging_cnnLSTM1{i}.h5') for i in range(n_models)]
    print('bagged models loaded')
    ensemble_predictions = soft_voting(bagged_models,RGB_X_test)

    y_pred=ensemble_predictions
    y_test=np.argmax(y_test, axis=1)
    categorical_accuracy = accuracy_score(y_test,y_pred)
    print(f" Accuracy = {categorical_accuracy}")
    report = classification_report(y_test, y_pred)
    print(f"Classification Report:\n{report}")
    cm = confusion_matrix(y_test, y_pred)
    print(cm) 
    
    
    