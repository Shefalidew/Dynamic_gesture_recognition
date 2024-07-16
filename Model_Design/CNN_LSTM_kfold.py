## model with reduced complexity and Dropout layers
import datetime
import numpy as np
import tensorflow as tf
from sklearn.model_selection import KFold
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, TimeDistributed, LSTM, Bidirectional, Dropout
from keras.models import Model
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping

#channels = int(input('Enter the number of channels:')) #We can use same model for both RGB and Depth separately
def keras_api_model2(X_train,channels):
    input_layer = Input(shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3],channels))
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
    lstm1 = Bidirectional(LSTM(256,return_sequences=True))(flat)
    drop4 = TimeDistributed(Dropout(0.3))(lstm1)
    lstm2 = Bidirectional(LSTM(256,return_sequences = True))(drop4)
    drop5 = TimeDistributed(Dropout(0.3))(lstm2)
    lstm3 = Bidirectional(LSTM(128))(drop5)
    dense1 = Dense(128,activation='relu')(lstm3)
    dense2 = Dense(64, activation='relu')(dense1)
    output = Dense(12,activation='softmax')(dense2)
    model = Model(input_layer,output)
    return model


if __name__ == '__main__':
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
    print(f'Y_train shape: {Y_train.shape}')
    X_test = np.load('Master_Thesis_codes/extended_BB_X_test_scaled.npy')
    RGB_X_test = X_test[:,:,:,:,:]
    print(f'RGB_X_test shape: {RGB_X_test.shape}')
    Y_test = np.load('Master_Thesis_codes/extended_BB_Y_test_scaled.npy')
    print(f'Y_test shape: {Y_test.shape}')
    X_val = np.load('Master_Thesis_codes/extended_BB_X_val_scaled.npy')
    RGB_X_val = X_val[:,:,:,:,:]
    print(f'RGB_X_val shape: {RGB_X_val.shape}')
    Y_val = np.load('Master_Thesis_codes/extended_BB_Y_val_scaled.npy')
    print(f'Y_val shape: {Y_val.shape}')
    print('Data loading finished')
    
    
    num_folds = 5
    kf = KFold(n_splits=num_folds,shuffle=True)
    learning_rates = [0.001, 0.0025, 0.005, 0.008 , 0.0008]
    #Iterate over each fold
    for fold, (train_index, val_index) in enumerate(kf.split(RGB_X_train,Y_train)):
        print(f"Fold {fold + 1}/{num_folds}")
        print(f"Learning Rate = {learning_rates[fold]}")
        X_train_fold, X_val_fold = RGB_X_train[train_index], RGB_X_train[val_index]
        Y_train_fold, Y_val_fold = Y_train[train_index], Y_train[val_index]

        #model call
        model = keras_api_model2(X_train_fold,4)
        opt = SGD(learning_rate = learning_rates[fold],momentum=0.9)
        model.compile(optimizer = opt ,loss = tf.keras.losses.CategoricalCrossentropy() ,metrics = ['categorical_accuracy','mae'])
        print('Model compiled')
        model.summary()

        # CREATE CALLBACKS
        checkpoint = tf.keras.callbacks.ModelCheckpoint(f'CNNLSTM_BB_1{fold}.h5',monitor='val_loss', verbose=2,save_best_only=True, mode='min')
        log_dir = "logs/CNNLSTM_BB_1_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = TensorBoard(log_dir =log_dir, histogram_freq =1)
        earlystop_callback = EarlyStopping(monitor = 'val_loss', patience = 5, min_delta = 0.001)
        history_3_1 = model.fit(X_train_fold,Y_train_fold,epochs=25,batch_size = 32,validation_data = (X_val_fold,Y_val_fold),verbose=2,callbacks=[checkpoint,tensorboard_callback])

        #Evaluate the model on the validation data for this fold
        categorical_accuracy = model.evaluate(X_val_fold,Y_val_fold)
        print(f"Validation Accuracy = {categorical_accuracy}, for fold={fold}")
        #model.save(f'new_cnn_lstm_extended_rgb_trained_model{fold}')
#training stage finishes here
   

    
    
    
    
    
   