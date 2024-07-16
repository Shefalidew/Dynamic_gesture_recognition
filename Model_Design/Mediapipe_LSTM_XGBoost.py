import xgboost as xgb
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import accuracy_score, classification_report,confusion_matrix 
from keras.layers import Input, Dense, LSTM, Bidirectional, Dropout
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
    
    
    X_train = np.load('Master_Thesis_codes/BB/replaced_BB_landmark_X_train_scaled.npy',allow_pickle=True)
    print((X_train.shape))
    X_train = X_train.reshape(X_train.shape[0],X_train.shape[1],X_train.shape[2]*X_train.shape[3])
    print(X_train.shape)
    
    Y_train = np.load('Master_Thesis_codes/BB/extended_BB_Y_train.npy')
    Y_train = np.argmax(Y_train,axis=1)
    print(Y_train.shape)
    
    X_val = np.load('Master_Thesis_codes/BB/replaced_BB_landmark_X_val_scaled.npy',allow_pickle=True)
    print((X_val.shape))
    X_val = X_val.reshape(X_val.shape[0],X_val.shape[1],X_val.shape[2]*X_val.shape[3])
    print(X_val.shape)
    
    Y_val = np.load('Master_Thesis_codes/BB/extended_BB_Y_val.npy')
    Y_val = np.argmax(Y_val,axis=1)
    print(Y_val.shape)
    
    X_test = np.load('Master_Thesis_codes/BB/replaced_BB_landmark_X_test_scaled.npy',allow_pickle=True)
    print((X_test.shape))
    X_test = X_test.reshape(X_test.shape[0],X_test.shape[1],X_test.shape[2]*X_test.shape[3])
    print(X_test.shape)
    
    Y_test = np.load('Master_Thesis_codes/BB/extended_BB_Y_test.npy')
    Y_test = np.argmax(Y_test,axis=1)
    print(Y_test.shape)
    
    
    input_layer = Input(shape=(X_train.shape[1],X_train.shape[2]))
    lstm1 = Bidirectional(LSTM(256,return_sequences=True,activation='relu'))(input_layer)
    lstm2 = Bidirectional(LSTM(256,return_sequences=True,activation='relu'))(lstm1)
    lstm3 = Bidirectional(LSTM(128))(lstm2)
    dense1 = Dense(128,activation='relu')(lstm3)
    dense2 = Dense(64,activation='relu')(dense1)
    output = Dense(12,activation='softmax')(dense2)
    
    new_model = tf.keras.models.load_model('mediapipe_kfold_BB_replacedZ_2.h5')
    #print('Loaded replacedZ_mediapipe_model trained one')
    #intermediate_layer_model = Model(inputs=input_layer, outputs=lstm3)
    intermediate_layer_model = Model(inputs=new_model.get_layer('bidirectional_6').input, outputs=new_model.get_layer('bidirectional_8').output)
    train_features = intermediate_layer_model.predict(X_train)
    print(f'features extracted:{train_features.shape}')
    
    reshaped_tensor = tf.reshape(train_features, [tf.shape(train_features)[0], -1])
    print(reshaped_tensor.shape)
    # Step 2: Create a DataFrame (Optional)
    df = pd.DataFrame(reshaped_tensor)
    train_features = df
    print(train_features.shape)    
    
    xgb_model = xgb.XGBClassifier(num_class=12, objective ='multi:softmax',tree_method='hist',random_state=42,learning_rate=0.15910942277102558, max_depth= 15, n_estimators= 106, subsample=0.6897744621575949)
    print(xgb_model)
    history = xgb_model.fit(train_features, Y_train)
    xgb_model.save_model('XGBoost_model_BB_mediapipeLSTM3.json')
    
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