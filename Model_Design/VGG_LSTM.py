
import numpy as np
import datetime
from keras.callbacks import EarlyStopping, TensorBoard,ReduceLROnPlateau
from keras.layers import Dense,  MaxPool2D, Flatten, Input, TimeDistributed, LSTM, Bidirectional, Dropout,Reshape
from keras.models import Model
from keras.optimizers import SGD
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.applications import VGG16
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix


if __name__ == '__main__':
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices)) 
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    print('Transfer Learning using VGG16 on RGB images')
    X_train = np.load('Master_Thesis_codes/extended_BB_X_train_scaled.npy')
    RGB_X_train = X_train[:,:,:,:,0:3]
    print(f'X_train shape: {RGB_X_train.shape}')
    X_test = np.load('Master_Thesis_codes/extended_BB_X_test_scaled.npy')
    RGB_X_test = X_test[:,:,:,:,0:3]
    print(f'X_test shape: {RGB_X_test.shape}')
    X_val = np.load('Master_Thesis_codes/extended_BB_X_val_scaled.npy')
    RGB_X_val = X_val[:,:,:,:,0:3]
    print(f'X_val shape: {RGB_X_val.shape}')
    
    Y_train = np.load('Master_Thesis_codes/extended_BB_Y_train_scaled.npy')
    print(f'Y_train shape: {Y_train.shape}')
    Y_test = np.load('Master_Thesis_codes/extended_BB_Y_test_scaled.npy')
    Y_val = np.load('Master_Thesis_codes/extended_BB_Y_val_scaled.npy')
   
    print('Data loading finished')
    
    # Load pre-trained VGG16 model with ImageNet weights
    base_model = VGG16(weights='imagenet', include_top=False)
    # Freeze pre-trained layers
    #for layer in base_model.layers:
        #layer.trainable = False
    base_model.trainable = False

    inputs = Input(shape=(RGB_X_train.shape[1],RGB_X_train.shape[2],RGB_X_train.shape[3],RGB_X_train.shape[-1]))
    y = Reshape((RGB_X_train.shape[1]*RGB_X_train.shape[2],RGB_X_train.shape[3],3))(inputs)
    print('Reshaped Input layer:',y.shape)
    #x = tf.repeat(inputs,3,axis=-1) # Repeat single-channel depth data across RGB channels
    x = base_model(y,training=False)
    
    x = TimeDistributed(Flatten())(x)
    x = Bidirectional(LSTM(256,return_sequences=True))(x)
    x = TimeDistributed(Dropout(0.3))(x)
    x = Bidirectional(LSTM(256,return_sequences = True))(x)
    x = TimeDistributed(Dropout(0.3))(x)
    x = Bidirectional(LSTM(128,return_sequences = True))(x)
    x = TimeDistributed(Dropout(0.3))(x)
    x = Bidirectional(LSTM(64))(x)
    x = Dense(128,activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(12,activation='softmax')(x)
    model = Model(inputs, outputs)


    opt = SGD(learning_rate=0.005,momentum=0.9) #parameters suggested by He [1]
    model.compile(optimizer = opt,loss='categorical_crossentropy', metrics=["categorical_accuracy"]) 
    model.summary()
    # CREATE CALLBACKS
    checkpoint = tf.keras.callbacks.ModelCheckpoint('VGG16_BB_rgbmodel_TL.h5',monitor='val_loss', verbose=2,save_best_only=True, mode='min')
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.6,patience=5, min_lr=0.00001)
    log_dir = "logs/VGG16_BB_rgbmodel_TL" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir =log_dir, histogram_freq =1)

    history = model.fit(RGB_X_train,Y_train, epochs=100,batch_size = 12,validation_data = (RGB_X_val,Y_val),verbose = 2,callbacks=[checkpoint,reduce_lr,tensorboard_callback])
    #model.save('VGG16_rgbmodel1_TL')
    #Evaluate the model on the validation data for this fold
    y_test = np.argmax(Y_test,axis=1)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred,axis=1)
    print("Predictions:", y_pred)
    
    #Evaluate the model on the validation data for this fold
    categorical_accuracy = accuracy_score(y_test,y_pred)
    print(f" Accuracy = {categorical_accuracy}")
    report = classification_report(y_test, y_pred)
    print(f"Classification Report:\n{report}")
    cm = confusion_matrix(y_test, y_pred)
    print(cm) 

    