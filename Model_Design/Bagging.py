## model with reduced complexity and Dropout layers
import numpy as np
import datetime
import tensorflow as tf
from sklearn.utils import resample
from keras import layers, models
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, TimeDistributed, LSTM, Bidirectional, Dropout
from keras.models import Model, load_model
from keras.optimizers import SGD
from keras.callbacks import ReduceLROnPlateau, TensorBoard, EarlyStopping
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix 

#channels = int(input('Enter the number of channels:')) #We can use same model for both RGB and Depth separately
def keras_api_model2(X_train):
    input_layer = Input(shape=(X_train.shape[1],X_train.shape[2],X_train.shape[3],4))
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

def train_model(X_train,y_train,i):
    model = keras_api_model2(X_train)
    opt = SGD(learning_rate=0.0025,momentum=0.9)
    model.compile(optimizer = opt ,loss = tf.keras.losses.CategoricalCrossentropy() ,metrics = ['categorical_accuracy','mae'])
    print('Model compiled')
    model.summary()
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f'bagging_BB_cnnlstm{i}.h5',monitor='categorical_accuracy', verbose=2,save_best_only=True, mode='max')
    log_dir = "logs/bagging_BB_cnnlstm" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = TensorBoard(log_dir =log_dir, histogram_freq =1)
    history = model.fit(X_train,y_train, epochs=20,batch_size=32,verbose = 2,callbacks=[checkpoint,tensorboard_callback])
    return model
    
def bagging_ensemble(X,y,n_models=5,bag_fraction = 0.8):
    models = []
    for i in range(n_models):
        X_bag,y_bag = resample(X,y,replace = True, n_samples=int(bag_fraction*len(X)))
        #labels already one-hot encoded
        model = train_model(X_bag,y_bag,i)
        #model.save(f'bagging_BB_cnnlstm{i}.h5')
        models.append(model)
    return models

def majority_voting_ensemble(models,X_test):
    predictions = []
    for model in models:
        predictions.append(np.argmax(model.predict(X_test),axis=1))
    print(predictions)
    
    ensemble_preds = np.zeros((len(predictions[0])))
    print(f'ensemble_preds = {ensemble_preds.shape}')
    for i in range(len(predictions[0])):
        votes = np.stack([preds[i] for preds in predictions],axis =0)
        print('votes:',votes)
        ensemble_preds[i] = np.argmax(np.bincount(votes))
        print(ensemble_preds[i])
        print('Actual :',y_test[i])
    print(ensemble_preds.shape)
    return ensemble_preds

if __name__ =='__main__':
    print('Performing bootstrap aggregation to train CNN+LSTM models on bootstraped subsets of data')
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
    
    #bagged_models = bagging_ensemble(RGB_X_train,Y_train,n_models = 5)
    
    print('Majority voting ensmeble prediction')
    n_models = 10
    num_classes = 12
    model_1 = load_model('Trained_models/CNNLSTM_BB_10.h5')
    model_2 = load_model('Trained_models/CNNLSTM_BB_11.h5')
    model_3 = load_model('Trained_models/CNNLSTM_BB_12.h5')
    model_4 = load_model('Trained_models/CNNLSTM_BB_13.h5')
    model_5 = load_model('Trained_models/CNNLSTM_BB_14.h5')
    model_6 = load_model('Trained_models/bagging_BB_cnnlstm0.h5')
    model_7 = load_model('Trained_models/bagging_BB_cnnlstm1.h5')
    model_8 = load_model('Trained_models/bagging_BB_cnnlstm2.h5')
    model_9 = load_model('Trained_models/bagging_BB_cnnlstm3.h5')
    model_10 = load_model('Trained_models/bagging_BB_cnnlstm4.h5')
    bagged_models = [model_1,model_2,model_3,model_4,model_5,model_6,model_7,model_8,model_9,model_10]
    #bagged_models = [load_model(f'bagging_cnnLSTM1{i}.h5') for i in range(n_models)]
    print('bagged models loaded')
    y_test=np.argmax(Y_test, axis=1)
    ensemble_predictions = majority_voting_ensemble(bagged_models,RGB_X_test)
    
    y_pred = ensemble_predictions
    
    categorical_accuracy = accuracy_score(y_test,y_pred)
    print(f" Accuracy = {categorical_accuracy}")
    report = classification_report(y_test, y_pred)
    print(f"Classification Report:\n{report}")
    cm = confusion_matrix(y_test, y_pred)
    print(cm) 
    
    