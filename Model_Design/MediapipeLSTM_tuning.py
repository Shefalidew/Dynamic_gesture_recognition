
import numpy as np
import datetime
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix 
from keras.models import Sequential
from keras import layers
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, TensorBoard,TerminateOnNaN
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
import keras_tuner

def build_model(hp):
    units = hp.Int("units", min_value=32, max_value=512, step=32)
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Boolean("dropout")
    lr = hp.Float("lr", min_value=1e-4, max_value=5e-3, sampling="log")
    # call existing model-building code with the hyperparameter values.
    model = lstm_model(units=units, activation=activation, dropout=dropout, lr=lr)
    return model	
    
def lstm_model(units,activation,dropout,lr):
    model = Sequential()
    model.add(layers.Bidirectional(layers.LSTM(units = units, activation = 'relu',return_sequences =True),input_shape=(X_train.shape[1],X_train.shape[2])))
    if dropout:
        model.add(layers.Dropout(rate=0.4))
    model.add(layers.Bidirectional(layers.LSTM(units = units, activation = 'relu',return_sequences =True)))
    if dropout:
        model.add(layers.Dropout(rate=0.4))
    model.add(layers.Bidirectional(layers.LSTM(units = units, activation = 'relu',return_sequences =False)))
    model.add(layers.Dense(units = units, activation = activation))
    model.add(layers.Dense(units = units, activation = activation))
    if dropout:
        model.add(layers.Dropout(rate=0.4))
    model.add(layers.Dense(12, activation="softmax"))
    model.compile(optimizer=SGD(learning_rate =lr),loss="categorical_crossentropy", metrics = ["accuracy","mae"],)
    print('Model compiled')    
    return model
    
def train_model(X_train,y_train,X_val,y_val):
    ##Instantiating the tuner
    build_model(keras_tuner.HyperParameters())
    tuner = keras_tuner.RandomSearch(hypermodel=build_model,objective="val_loss",max_trials=20,executions_per_trial=2,overwrite=True,directory="my_dir",project_name="tuning_trials_BB_mediapipeLSTM")
    tuner.search_space_summary()
    
    log_dir = "logs/mediapipeLSTM_BB_tuning1" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir =log_dir, histogram_freq =1)
    terminate_callback = TerminateOnNaN()
    tuner.search(X_train, y_train, epochs=100,batch_size=32, validation_data=(X_val, y_val),callbacks=[tensorboard_callback,terminate_callback])
    
    models = tuner.get_best_models(num_models=1)
    best_model = models[0]
    best_model.summary()
    tuner.results_summary()
    
    best_hps = tuner.get_best_hyperparameters(3)
    print(best_hps)
    # Build the model with the best hp.
    model = build_model(best_hps[0])
    
    print('Now starting the training of the best model')
    history = model.fit(X_train,y_train, epochs=50,batch_size=12,validation_data = (X_val,y_val),verbose = 2,callbacks=[tensorboard_callback,terminate_callback])
    model.summary()
    model.save('tuned_BB_mediapipeLSTM1.h5')
    return model    


if __name__ =='__main__':
    print('MediapipeLSTM classifier for selected landmark dataset where landmark.z is replaced with actual depth values')
    print('With Best parameter values found using Randomized Search using Keras Tuners')
    # GPU Initialization
    physical_devices = tf.config.list_physical_devices('GPU') 
    print("Num GPUs:", len(physical_devices)) 
    config = ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.9
    config.gpu_options.allow_growth = True
    session = InteractiveSession(config=config)
    
    X_train = np.load('Master_Thesis_codes/selected_landmark_BB_X_train.npy',allow_pickle=True)
    print((X_train.shape))
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2]*X_train.shape[3])
    print(X_train.shape)
    
    Y_train = np.load('Master_Thesis_codes/extended_BB_Y_train_scaled.npy')
    print(Y_train.shape)
    
    X_val = np.load('Master_Thesis_codes/selected_landmark_BB_X_val.npy',allow_pickle=True)
    print((X_val.shape))
    X_val = X_val.reshape(X_val.shape[0],X_val.shape[1],X_val.shape[2]*X_val.shape[3])
    print(X_val.shape)
    
    Y_val = np.load('Master_Thesis_codes/extended_BB_Y_val_scaled.npy')
    print(Y_val.shape)
    
    X_test = np.load('Master_Thesis_codes/selected_landmark_BB_X_test.npy',allow_pickle=True)
    print((X_test.shape))
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2]*X_test.shape[3])
    print(X_test.shape)
    
    Y_test = np.load('Master_Thesis_codes/extended_BB_Y_test_scaled.npy')
    
    print(Y_test.shape)

    
    model = train_model(X_train,Y_train,X_val,Y_val)
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred,axis=1)
    y_test = np.argmax(Y_test,axis=1)
    #Evaluate the model on the validation data for this fold
    categorical_accuracy = accuracy_score(y_test,y_pred)
    print(f" Accuracy = {categorical_accuracy}")
    report = classification_report(y_test, y_pred)
    print(f"Classification Report:\n{report}")
    cm = confusion_matrix(y_test, y_pred)
    print(cm) 