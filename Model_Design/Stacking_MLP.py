
import mediapipe as mp
import numpy as np
import pandas as pd
import datetime
import tensorflow as tf
import xgboost as xgb
from keras.models import Sequential
from keras import layers
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.layers import Input, Dense, Dropout
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.callbacks import TensorBoard,TerminateOnNaN,ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix
from sklearn.preprocessing import MinMaxScaler
import keras_tuner
from keras import regularizers

	

def keras_api_model(X_train):
    input_layer = Input(shape=(X_train.shape[1]))
    dense1 = Dense(128,activation='relu',kernel_regularizer=regularizers.L1(0.01),
                     activity_regularizer=regularizers.L2(0.01))(input_layer)
    #drop1 = Dropout(0.5)(dense1)
    dense2 = Dense(64, activation='relu',kernel_regularizer=regularizers.L1(0.01),
                     activity_regularizer=regularizers.L2(0.01))(dense1)
    dense3 = Dense(32, activation='relu',kernel_regularizer=regularizers.L1(0.01),
                     activity_regularizer=regularizers.L2(0.01))(dense2)
    drop2 = Dropout(0.5)(dense3)
    output = Dense(12,activation='softmax')(drop2)
    model = Model(input_layer,output)
    return model
    
def build_model(hp):
    units = hp.Int("units", min_value=32, max_value=256, step=32)
    activation = hp.Choice("activation", ["relu", "tanh"])
    dropout = hp.Boolean("dropout")
    lr = hp.Float("lr", min_value=1e-4, max_value=5e-3, sampling="log")
    # call existing model-building code with the hyperparameter values.
    model = keras_api_model2(units=units, activation=activation, dropout=dropout, lr=lr)
    return model	

def keras_api_model2(units,activation,dropout,lr):
    model = Sequential()
    model.add(layers.Dense(units = units, activation = activation))
    if dropout:
        model.add(layers.Dropout(rate=0.4))
    model.add(layers.Dense(12, activation="softmax"))
    model.compile(optimizer=SGD(learning_rate =lr),loss="categorical_crossentropy", metrics = ["accuracy","mae"],)
    print('Model compiled')
    return model

def train_model2(X_train,y_train,X_val,y_val):
    ##Instantiating the tuner
    build_model(keras_tuner.HyperParameters())
    tuner = keras_tuner.RandomSearch(hypermodel=build_model,objective="val_loss",max_trials=20,executions_per_trial=2,overwrite=True,directory="my_dir",project_name="tuning_trials")
    tuner.search_space_summary()
    
    log_dir = "logs/stacking_meta_MLP_tuning" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir =log_dir, histogram_freq =1)
    tuner.search(X_train, y_train, epochs=100,batch_size=32, validation_data=(X_val, y_val),callbacks=[tensorboard_callback])
    
    models = tuner.get_best_models(num_models=1)
    best_model = models[0]
    #best_model.summary()
    tuner.results_summary()
    
    best_hps = tuner.get_best_hyperparameters(3)
    print(best_hps)
    # Build the model with the best hp.
    model = build_model(best_hps[0])
    
    print('Now starting the training of the best model')
    history = model.fit(X_train,y_train, epochs=100,batch_size=12,validation_data = (X_val,y_val),verbose = 2,callbacks=[tensorboard_callback])
    model.summary()
    model.save('Meta_MLP_model_tuned.h5')
    return model


def train_model(X_train,y_train,X_val,y_val):
    model = keras_api_model(X_train)
    opt = SGD(learning_rate=0.00025,momentum=0.9)
    model.compile(optimizer = opt ,loss = tf.keras.losses.CategoricalCrossentropy() ,metrics = ['categorical_accuracy','mae'])
    print('Model compiled')
    model.summary()
    checkpoint = tf.keras.callbacks.ModelCheckpoint('Meta_MLP_model_BB_normalized.h5',monitor='val_loss', verbose=2,save_best_only=True, mode='min')
    #reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.8,patience=5, min_lr=0.0001)
    log_dir = "logs/stacking_meta_MLP_normalized" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir =log_dir, histogram_freq =1)

    history = model.fit(X_train,y_train, epochs=100,batch_size=32,validation_data = (X_val,y_val),verbose = 2,callbacks=[tensorboard_callback,checkpoint])
    #model.save('Meta_MLP_model_BB_normalized.h5')
    return model

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
    
    X_val_scaled = np.load('Master_Thesis_codes/BB/extended_BB_X_val_scaled.npy')
    print((X_val_scaled.shape))
    RGB_X_val = X_val_scaled[:,:,:,:,:]
    print(f'RGB_X_val shape: {RGB_X_val.shape}')
    
    mediapipe_X_val_array = np.load('Master_Thesis_codes/BB/selected_landmark_BB_X_val.npy',allow_pickle=True)
    print(f'mediapipe_X_val shape : {mediapipe_X_val_array.shape}')
    mediapipe_X_val = mediapipe_X_val_array.reshape(mediapipe_X_val_array.shape[0],mediapipe_X_val_array.shape[1],mediapipe_X_val_array.shape[2]*mediapipe_X_val_array.shape[3])
    print(mediapipe_X_val.shape)
    
    y_val = np.load('Master_Thesis_codes/BB/extended_BB_Y_val_scaled.npy')
    print(y_val.shape)
    
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
    
    ##Normalization
    scaler = MinMaxScaler(feature_range=(0,1))
    
    ##loading the trained models
    mediapipeLSTM = load_model('selected_BB_mediapipeModel4.h5')
    cnnLSTM = load_model('Working_trained_models/CNNLSTM_BB_trained.h5')
    mediapipeXGB = xgb.Booster(model_file='XGBoost_model_with_CNNLSTM2.json')
    
    ##Need to extract features from mediapipe+LSTM model to be given as input to XGBoostClassifier
    intermediate_layer_model = Model(inputs=cnnLSTM.get_layer('input_1').input, outputs=cnnLSTM.get_layer('bidirectional_2').output)
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
    val_features = intermediate_layer_model.predict(RGB_X_val)
    reshaped_tensor = tf.reshape(val_features, [tf.shape(val_features)[0], -1])
    print(reshaped_tensor.shape)
    # Step 2: Create a DataFrame (Optional)
    df = pd.DataFrame(reshaped_tensor)
    val_features = df
    print(val_features.shape)
    
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
    merged = meta_train
    print(f'Shape of merged outputs:{merged.shape}')
    original_shape = merged.shape  # Save the original shape
    reshaped_data = merged.reshape(-1, merged.shape[-1])  # Reshape to 2D
    # Apply MinMaxScaler
    scaled_data = scaler.fit_transform(reshaped_data.astype(np.float16))
    merged_scaled_data = scaled_data.reshape(original_shape)
    print('merged_scaled:',merged_scaled_data)
    
    meta_val = np.zeros((len(RGB_X_val), len(base_models)))
    pred = base_models[0].predict(RGB_X_val)
    pred = np.argmax(pred,axis=1)
    meta_val[:,0] = pred #Prediction value of cnnLSTM
    pred = base_models[1].predict(mediapipe_X_val)
    pred = np.argmax(pred,axis=1)
    meta_val[:,1] = pred #Prediction value of mediapipeLSTM
    meta_val[:,2] = base_models[2].predict(xgb.DMatrix(val_features)) #Prediction value of mediapipeXGB
    print(meta_val[:,2].shape)
    val = meta_val
    print(f'Shape of merged outputs:{val.shape}')
    original_shape = val.shape  # Save the original shape
    reshaped_data = val.reshape(-1, val.shape[-1])  # Reshape to 2D
    # Apply MinMaxScaler
    scaled_data = scaler.fit_transform(reshaped_data.astype(np.float16))
    val_scaled_data = scaled_data.reshape(original_shape)
    print('val_scaled:',val_scaled_data)
   
    meta_model = train_model(merged_scaled_data, y_train,val_scaled_data,y_val)
    
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
    test_data = meta_test
    print(f'Shape of merged outputs:{test_data.shape}')
    original_shape = test_data.shape  # Save the original shape
    reshaped_data = test_data.reshape(-1, test_data.shape[-1])  # Reshape to 2D
    # Apply MinMaxScaler
    scaled_data = scaler.fit_transform(reshaped_data.astype(np.float16))
    test_scaled_data = scaled_data.reshape(original_shape)
    print('test_scaled:',test_scaled_data)
    
    
    y_pred = meta_model.predict(test_scaled_data)
    y_pred = np.argmax(y_pred,axis=1)
    y_test = np.argmax(y_test,axis=1)
    
    print("Predictions:", y_pred)
     
    #Evaluate the model on the validation data for this fold
    categorical_accuracy = accuracy_score(y_test,y_pred)
    print(f" Accuracy = {categorical_accuracy}")
    report = classification_report(y_test, y_pred)
    print(f"Classification Report:\n{report}")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    
    
    
    
    
    
    