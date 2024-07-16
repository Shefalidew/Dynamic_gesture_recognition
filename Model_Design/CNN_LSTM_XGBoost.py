import xgboost as xgb
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix 
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, TimeDistributed, LSTM, Bidirectional,Conv3D,MaxPooling3D, Dropout
from keras.models import Model
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
    
    X_train = np.load('Master_Thesis_codes/extended_BB_X_train_scaled.npy')
    RGB_X_train = X_train[:,:,:,:,:]
    print(f'RGB_X_train shape: {RGB_X_train.shape}')
    Y_train = np.load('Master_Thesis_codes/extended_BB_Y_train_scaled.npy')
    Y_train = np.argmax(Y_train,axis=1)
    print(f'Y_train shape: {Y_train.shape}')
    X_test = np.load('Master_Thesis_codes/extended_BB_X_test_scaled.npy')
    RGB_X_test = X_test[:,:,:,:,:]
    print(f'RGB_X_test shape: {RGB_X_test.shape}')
    Y_test = np.load('Master_Thesis_codes/extended_BB_Y_test_scaled.npy')
    Y_test = np.argmax(Y_test,axis=1)
    print(f'Y_test shape: {Y_test.shape}')
    X_val = np.load('Master_Thesis_codes/extended_BB_X_val_scaled.npy')
    RGB_X_val = X_val[:,:,:,:,:]
    print(f'RGB_X_val shape: {RGB_X_val.shape}')
    Y_val = np.load('Master_Thesis_codes/extended_BB_Y_val_scaled.npy')
    print(f'Y_val shape: {Y_val.shape}')
    print('Data loading finished')
    
    #either write the CNN+LSTM architecture here
    input_layer = Input(shape=(RGB_X_train.shape[1],RGB_X_train.shape[2],RGB_X_train.shape[3],4))
    conv1 = TimeDistributed(Conv2D(8,kernel_size=(3,3),activation='relu'))(input_layer)
    conv2 = TimeDistributed(Conv2D(8,kernel_size=(3,3),activation='relu'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(conv2)
    conv3 = TimeDistributed(Conv2D(16,kernel_size=(3,3),activation='relu'))(pool1)
    conv4 = TimeDistributed(Conv2D(16,kernel_size=(3,3),activation='relu'))(conv3)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(conv4)
    conv5 = TimeDistributed(Conv2D(32,kernel_size=(3,3),activation='relu'))(pool2)
    conv6 = TimeDistributed(Conv2D(32,kernel_size=(3,3),activation='relu'))(conv5)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(conv6)
    flat = TimeDistributed(Flatten())(pool3)
    lstm1 = Bidirectional(LSTM(256,return_sequences=True))(flat)
    lstm2 = Bidirectional(LSTM(128,return_sequences=True))(lstm1)
    lstm3 = Bidirectional(LSTM(64))(lstm2)
    dense1 = Dense(128,activation='relu')(lstm3)
    dense2 = Dense(64,activation='relu')(dense1)
    output = Dense(12,activation='softmax')(dense2)
    
    #or load the CNN+LSTM trained model
    new_model = tf.keras.models.load_model('Working_trained_models/CNNLSTM_BB_trained.h5')
    #for the created architecture
    #intermediate_layer_model = Model(inputs=input_layer, outputs=lstm3)
    #for the loaded model
    intermediate_layer_model = Model(inputs=new_model.get_layer('input_1').input,outputs=new_model.get_layer('bidirectional_2').output)
    train_features = intermediate_layer_model.predict(RGB_X_train)
    print(f'features extracted:{train_features.shape}')
    
    reshaped_tensor = tf.reshape(train_features, [tf.shape(train_features)[0], -1])
    print(reshaped_tensor.shape)
    # Step 2: Create a DataFrame (Optional)
    df = pd.DataFrame(reshaped_tensor)
    train_features = df
    print(train_features.shape)
    
    xgb_model = xgb.XGBClassifier(num_class=12, objective ='multi:softmax',tree_method='hist',random_state=42,learning_rate=0.09555846116602244, max_depth= 8, n_estimators= 283, subsample=0.5488472640974592)
    print(xgb_model)
    history = xgb_model.fit(train_features, Y_train)
    xgb_model.save_model('XGBoost_model_with_CNNLSTM2.json')
    
    test_features = intermediate_layer_model.predict(X_test)
    y_pred = xgb_model.predict(test_features)
    print("Predictions:", y_pred)
     
    #Evaluate the model on the validation data for this fold
    categorical_accuracy = accuracy_score(Y_test,y_pred)
    print(f" Accuracy = {categorical_accuracy}")
    report = classification_report(Y_test, y_pred)
    print(f"Classification Report:\n{report}")
    cm = confusion_matrix(Y_test, y_pred)
    print(cm) 