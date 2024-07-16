import numpy as np
import datetime
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, TimeDistributed, LSTM, Bidirectional,Dropout,Concatenate
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix 

def feature_level_fusion_model(input_layer):
    conv1 = TimeDistributed(Conv2D(8,kernel_size=(3,3),activation='relu'))(input_layer)
    conv2 = TimeDistributed(Conv2D(8,kernel_size=(3,3),activation='relu'))(conv1)
    pool1 = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(conv2)
    drop1 = TimeDistributed(Dropout(0.3))(pool1)
    conv3 = TimeDistributed(Conv2D(16,kernel_size=(3,3),activation='relu'))(drop1)
    conv4 = TimeDistributed(Conv2D(16,kernel_size=(3,3),activation='relu'))(conv3)
    pool2 = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(conv4)
    drop2 = TimeDistributed(Dropout(0.3))(pool2)
    conv5 = TimeDistributed(Conv2D(32,kernel_size=(3,3),activation='relu'))(drop2)
    conv6 = TimeDistributed(Conv2D(32,kernel_size=(3,3),activation='relu'))(conv5)
    pool3 = TimeDistributed(MaxPooling2D(pool_size=(2,2)))(conv6)
    drop3 = TimeDistributed(Dropout(0.3))(pool3)
    flat = TimeDistributed(Flatten())(drop3)
    return flat

def model_training(features):
    lstm1 = Bidirectional(LSTM(256,return_sequences=True))(features)
    drop4 = TimeDistributed(Dropout(0.3))(lstm1)
    lstm2 = Bidirectional(LSTM(256,return_sequences = True))(drop4)
    drop5 = TimeDistributed(Dropout(0.3))(lstm2)
    lstm3 = Bidirectional(LSTM(128))(drop5)
    dense1 = Dense(128,activation='relu')(lstm3)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(12,activation='softmax')(dense2)
    return output
    
    
    
if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices)) 
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.5
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    X_train = np.load('Master_Thesis_codes/extended_BB_X_train_scaled.npy')
    RGB_X_train = X_train[:,:,:,:,0:3]
    Depth_X_train = X_train[:,:,:,:,-1]
    print(f'RGB_X_train shape: {RGB_X_train.shape}')
    X_test = np.load('Master_Thesis_codes/extended_BB_X_test_scaled.npy')
    RGB_X_test = X_test[:,:,:,:,0:3]
    Depth_X_test = X_test[:,:,:,:,-1]
    print(f'RGB_X_test shape: {RGB_X_test.shape}')
    X_val = np.load('Master_Thesis_codes/extended_BB_X_val_scaled.npy')
    RGB_X_val = X_val[:,:,:,:,0:3]
    Depth_X_val = X_val[:,:,:,:,-1]
    print(f'RGB_X_val shape: {RGB_X_val.shape}')
    Y_train = np.load('Master_Thesis_codes/extended_BB_Y_train_scaled.npy')
    print(f'Y_train shape: {Y_train.shape}')
    Y_test = np.load('Master_Thesis_codes/extended_BB_Y_test_scaled.npy')
    print(f'Y_test shape: {Y_test.shape}')
    Y_val = np.load('Master_Thesis_codes/extended_BB_Y_val_scaled.npy')
    print(f'Y_val shape: {Y_val.shape}')

    print('Data loading finished')

    RGB_input = RGB_X_train
    rgb_shape = RGB_input.shape
    input_rgb = Input(shape=(rgb_shape[1],rgb_shape[2],rgb_shape[3],3))
    Depth_input = Depth_X_train
    depth_shape = Depth_input.shape
    input_depth = Input(shape = (depth_shape[1],depth_shape[2],depth_shape[3],1))
    

    RGB_feature = feature_level_fusion_model(input_rgb)
    print(RGB_feature.shape)
    Depth_feature = feature_level_fusion_model(input_depth)
    print(Depth_feature.shape)

    merged_features = Concatenate(axis=1)([RGB_feature,Depth_feature])
    print(merged_features.shape)

    output = model_training(merged_features)

    model = Model(inputs = [input_rgb,input_depth],outputs = output)
    opt = SGD(learning_rate=0.0025,momentum=0.9)
    model.compile(optimizer = opt ,loss = tf.keras.losses.CategoricalCrossentropy() ,metrics = ['categorical_accuracy','mae'])
    print('Model compiled')
    model.summary()
    
    # CREATE CALLBACKS
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'feature_fusion_BB_trained.h5',monitor='val_loss', verbose=2,save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6, patience=2, min_lr=0.0005)
    log_dir = "logs/feature_fusion_BB_trained" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir =log_dir, histogram_freq =1)
    earlystop_callback = EarlyStopping(monitor = 'val_loss', patience = 5, min_delta = 0.001)
    history_3_1 = model.fit([RGB_X_train,Depth_X_train],Y_train,epochs=25,batch_size = 32, validation_data =([RGB_X_val,Depth_X_val],Y_val), verbose=2 , callbacks=[earlystop_callback,checkpoint,tensorboard_callback,reduce_lr])

    #Evaluate the model on the validation data for this fold
    categorical_accuracy = model.evaluate([RGB_X_test,Depth_X_test],Y_test)
    print(f"Validation Accuracy = {categorical_accuracy}")
    y_pred = model.predict([RGB_X_test,Depth_X_test])
    y_pred=np.argmax(y_pred, axis=1)
    y_test=np.argmax(Y_test, axis=1)
    categorical_accuracy = accuracy_score(y_test,y_pred)
    print(f" Accuracy = {categorical_accuracy}")
    report = classification_report(y_test, y_pred)
    print(f"Classification Report:\n{report}")
    cm = confusion_matrix(y_test, y_pred)
    print(cm) 
    
    #training stage finishes here
    
    