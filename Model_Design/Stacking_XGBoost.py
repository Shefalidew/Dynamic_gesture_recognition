
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.models import Model, load_model
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix

	

if __name__ =='__main__':
    # GPU Initialization
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices)) 
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    
    print('Starting loading the 5D tensor image data')
    X_train_scaled = np.load('Master_Thesis_codes/BB/extended_BB_X_train_scaled.npy')
    print((X_train_scaled.shape))
    RGB_X_train = X_train_scaled[:,:,:,:,:]
    print(f'RGB_X_train shape: {RGB_X_train.shape}')
    
    mediapipe_X_train_array = np.load('Master_Thesis_codes/BB/selected_landmark_BB_X_train.npy',allow_pickle=True)
    print(f'mediapipe_X_train shape : {mediapipe_X_train_array.shape}')
    mediapipe_X_train = mediapipe_X_train_array.reshape(mediapipe_X_train_array.shape[0],mediapipe_X_train_array.shape[1],mediapipe_X_train_array.shape[2]*mediapipe_X_train_array.shape[3])
    print(mediapipe_X_train.shape)
    
    y_train = np.load('Master_Thesis_codes/BB/extended_BB_Y_train_scaled.npy')
    print(y_train.shape)
    
    X_test_scaled = np.load('Master_Thesis_codes/BB/extended_BB_X_test_scaled.npy')
    print((X_test_scaled.shape))
    RGB_X_test = X_test_scaled[:,:,:,:,:]
    print(f'RGB_X_test shape: {RGB_X_test.shape}')
    
    mediapipe_X_test_array = np.load('Master_Thesis_codes/BB/selected_landmark_BB_X_test.npy',allow_pickle=True)
    print(f'mediapipe_X_test shape : {mediapipe_X_test_array.shape}')
    mediapipe_X_test = mediapipe_X_test_array.reshape(mediapipe_X_test_array.shape[0],mediapipe_X_test_array.shape[1],mediapipe_X_test_array.shape[2]*mediapipe_X_test_array.shape[3])
    print(mediapipe_X_test.shape)
    
    y_test = np.load('Master_Thesis_codes/BB/extended_BB_Y_test_scaled.npy')
    print(y_test.shape)
    print('Data loading finished')
    
    
    ##loading the trained models
    mediapipeLSTM = load_model('Working_trained_models/Inference/selected_BB_mediapipeModel4.h5')
    cnnLSTM = load_model('Working_trained_models/Inference/CNNLSTM_BB_trained.h5')
    mediapipeXGB = xgb.Booster(model_file='XGBoost_model_with_CNNLSTM2.json')
    
    ##Need to extract features from mediapipe+LSTM model to be given as input to XGBoostClassifier
    intermediate_layer_model = Model(inputs=cnnLSTM.get_layer('input_1').input,outputs=cnnLSTM.get_layer('bidirectional_2').output)
    train_features = intermediate_layer_model.predict(RGB_X_train)
    print(f'features extracted:{train_features.shape}')
    reshaped_tensor = tf.reshape(train_features, [tf.shape(train_features)[0], -1])
    print(reshaped_tensor.shape)
    # Step 2: Create a DataFrame (Optional)
    df = pd.DataFrame(reshaped_tensor)
    train_features = df
    print(train_features.shape)
    test_features = intermediate_layer_model.predict(RGB_X_test)
    reshaped_tensor = tf.reshape(test_features, [tf.shape(test_features)[0], -1])
    print(reshaped_tensor.shape)
    # Step 2: Create a DataFrame (Optional)
    df = pd.DataFrame(reshaped_tensor)
    test_features = df
    print(test_features.shape)
    
    ##Base models added as estimators
    base_models = []
    base_models.append(cnnLSTM)
    base_models.append(mediapipeLSTM)
    base_models.append(mediapipeXGB)
    
    ## Concatenating the predicitons of base learners to be given as input for meta learner
    meta_train = np.zeros((len(RGB_X_train), len(base_models)))
    pred = base_models[0].predict(RGB_X_train)
    pred = np.argmax(pred,axis=1)
    meta_train[:,0] = pred #Prediction value of cnnLSTM
    pred = base_models[1].predict(mediapipe_X_train)
    pred = np.argmax(pred,axis=1)
    meta_train[:,1] = pred #Prediction value of mediapipeLSTM
    meta_train[:,2] = base_models[2].predict(xgb.DMatrix(train_features)) #Prediction value of mediapipeXGB
    print(meta_train[:,2].shape)
    
    y_train = np.argmax(y_train,axis=1)
    print('changed shape of y_train',y_train.shape)
    
    merged = meta_train
    print(f'Shape of merged outputs:{merged.shape}')
    
    xgb_classifier = xgb.XGBClassifier(num_class=12, objective ='multi:softmax',tree_method='hist',random_state=42,learning_rate = 0.049480040452442135, max_depth = 25, n_estimators = 109, subsample = 0.8643979513372948)
    meta_model = xgb_classifier.fit(merged, y_train)
    
    meta_model.save_model('Meta_BB_learner_model.json')
    print('training of meta learner finished')
    
    ##Final prediction on X_test
    meta_test = np.zeros((len(RGB_X_test), len(base_models)))
    pred = base_models[0].predict(RGB_X_test)
    pred = np.argmax(pred,axis=1)
    meta_test[:,0] = pred #Prediction value of cnnLSTM
    pred = base_models[1].predict(mediapipe_X_test)
    pred = np.argmax(pred,axis=1)
    meta_test[:,1] = pred #Prediction value of mediapipeLSTM
    meta_test[:,2] = base_models[2].predict(xgb.DMatrix(test_features)) #Prediction value of mediapipeXGB
    print(meta_test[:,2].shape)
    
    y_test = np.argmax(y_test,axis=1)
    print('changed shape of y_test',y_test.shape)
        
    merged = meta_test
    print(f'Shape of merged outputs:{merged.shape}')
    y_pred = meta_model.predict(merged)
    
    print("Predictions:", y_pred)
     
    #Evaluate the model on the validation data for this fold
    categorical_accuracy = accuracy_score(y_test,y_pred)
    print(f" Accuracy = {categorical_accuracy}")
    report = classification_report(y_test, y_pred)
    print(f"Classification Report:\n{report}")
    cm = confusion_matrix(y_test, y_pred)
    print(cm) 
    
    
    
    
    