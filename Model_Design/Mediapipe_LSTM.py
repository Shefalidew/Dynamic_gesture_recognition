
import numpy as np
import datetime
import tensorflow as tf
from keras.optimizers import SGD
from keras.callbacks import TensorBoard,TerminateOnNaN,ReduceLROnPlateau, TensorBoard, EarlyStopping
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix





def build_model_bdlstm_1(x_train):
    model = Sequential()
    model.add(layers.Bidirectional(layers.LSTM(256, return_sequences=True,dropout = 0.3),
                                               input_shape=(x_train.shape[1], x_train.shape[2])))   
    #model.add(layers.Dropout(rate=0.4))
    model.add(layers.Bidirectional(layers.LSTM(256,return_sequences=True,dropout = 0.3)))
    #model.add(layers.Dropout(rate=0.4))
    model.add(layers.Bidirectional(layers.LSTM(128,return_sequences=False)))
    #model.add(layers.Dropout(rate=0.4))
    model.add(layers.Dense(256, activation='relu'))
    #model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(256, activation='relu'))
    model.add(layers.Dropout(rate=0.4))
    model.add(layers.Dense(12, activation='softmax'))
    #model.summary()
    return model



if __name__ =='__main__':
    print('With scaled selected landmark points obtained from BB region clipped from images as dataset')
    X_train = np.load('Master_Thesis_codes/BB/selected_landmark_BB_X_train.npy',allow_pickle=True)
    print((X_train.shape))
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2]*X_train.shape[3])
    print(X_train.shape)
    Y_train = np.load('Master_Thesis_codes/BB/extended_BB_Y_train.npy')
    print(Y_train.shape)
    
    X_val = np.load('Master_Thesis_codes/BB/selected_landmark_BB_X_val.npy',allow_pickle=True)
    print((X_val.shape))
    X_val = X_val.reshape(X_val.shape[0],X_val.shape[1],X_val.shape[2]*X_val.shape[3])
    print(X_val.shape)
    Y_val = np.load('Master_Thesis_codes/BB/extended_BB_Y_val.npy')
    print(Y_val.shape)
    
    X_test = np.load('Master_Thesis_codes/BB/selected_landmark_BB_X_test.npy',allow_pickle=True)
    print((X_test.shape))
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2]*X_test.shape[3])
    print(X_test.shape)
    Y_test = np.load('Master_Thesis_codes/BB/extended_BB_Y_test.npy')
    print(Y_test.shape)

    # GPU Initialization
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices)) 
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    model_3_1 = build_model_bdlstm_1(X_train)
    opt = SGD(learning_rate = 0.0025)
    model_3_1.compile(optimizer = opt ,loss = tf.keras.losses.CategoricalCrossentropy() ,metrics = ['categorical_accuracy','mae'])
    print('Model compiled')
    model_3_1.summary()
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,patience=5, min_lr=0.0001)
    checkpoint = tf.keras.callbacks.ModelCheckpoint('selected_BB_mediapipeModel5.h5',monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    log_dir = "logs/selected_BB_mediapipeModel5" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir =log_dir, histogram_freq =1)
    terminate_callback = TerminateOnNaN()
    history_3_1 = model_3_1.fit(X_train,Y_train,epochs=250,batch_size = 32,verbose=2,validation_data = (X_val,Y_val),callbacks=[checkpoint,tensorboard_callback,terminate_callback])

    y_test = np.argmax(Y_test,axis=1)
    y_pred = model_3_1.predict(X_test)
    y_pred = np.argmax(y_pred,axis=1)
    #Evaluate the model on the validation data for this fold
    categorical_accuracy = accuracy_score(y_test,y_pred)
    print(f" Accuracy = {categorical_accuracy}")
    report = classification_report(y_test, y_pred)
    print(f"Classification Report:\n{report}")
    cm = confusion_matrix(y_test, y_pred)
    print(cm) 
    #training stage finishes here
    
    
