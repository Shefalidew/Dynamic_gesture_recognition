##working code to create training and test datasets

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
import h5py


class DataRepository():

    def __init__(self, codes=[], verbose = False):
        self.__verbose = verbose
        self.x_train = None
        self.x_val = None
        self.x_test = None
        self.y_train = None
        self.y_val = None
        self.y_test = None
        self.dataPergesture =[]
        self.numClasses = 0
        self.codes=codes
        self.path = "/home/shefali/recorded_data_BB/"
        self.features, self.labels = self.__loadData(self.codes, updateClasses=True)
    

    def __loadData(self, codes, updateClasses=False):
        j=0
        try:
            for code in codes:
                dirname = os.path.join(self.path,code)
                self.listfile = os.listdir(dirname)
                self.listfile = sorted(self.listfile, key=str.casefold)
                print(self.listfile)
                if updateClasses:
                    self.numClasses = len(self.listfile)
                    #print(self.numClasses)
                for gesture in self.listfile:
                    self.gesture_path = os.path.join(dirname,gesture)
                    for file in os.listdir(self.gesture_path):
                        if file.endswith(".h5"):
                            filepath = os.path.join(self.gesture_path, file)
                            print(filepath)
                            hf = h5py.File(filepath, 'r')                  
                            content = []
                            for i in range(len(hf.keys())):
                                content.append(hf.get('frame_'+str(i)))
                            #content = content.reindex(list(range(0, frames)), fill_value=default)
                            #content.fillna(default, inplace = True)

                            content = np.array(content)
                            self.dataPergesture.append((gesture, content))
                            j=j+1
                            print('j:',j)
                            hf.close()
        except FileNotFoundError:
            print("File doesn't exist")
            pass
                
        features = [n[1] for n in self.dataPergesture]
        labels = [n[0] for n in self.dataPergesture]
        return features, labels

    def getDataAndLabels(self):
        #this function is used to create a one-hot-encoded labels
        features = [n[1] for n in self.dataPergesture]
        x = features
        lower_gestures = [x.lower() for x in self.listfile]  
        
        y = [label.lower() for label in self.labels]
        encoder = LabelBinarizer() 
        '''OneHotEncoder needs data in integer encoded form first 
        to convert into its respective encoding 
        which is not required in the case of LabelBinarizer.'''
        encoder.fit(lower_gestures)
        y = encoder.transform(y)
        return np.array(x), np.array(y)
    
    def getForTraining(self):
        x_train, x_val, y_train, y_val = train_test_split(self.features,self.labels, 
                                                          test_size=0.40, random_state=42, stratify=self.labels)
        x_val, x_test, y_val, y_test = train_test_split(x_val, y_val,test_size=0.50, random_state=42, stratify=y_val)
        if self.__verbose:
            self.__summary()
        lower_gestures = [x.lower() for x in self.listfile]

        y_train = [y.lower() for y in y_train]
        y_val = [y.lower() for y in y_val]
        y_test = [y.lower() for y in y_test]

        encoder = LabelBinarizer()
        test = encoder.fit_transform(lower_gestures)

        y_train = encoder.transform(y_train)
        y_val = encoder.transform(y_val)
        y_test = encoder.transform(y_test)
        # Making numpy arrays
        self.x_train=np.array(x_train)
        self.y_train=np.array(y_train)
        self.x_val=np.array(x_val)
        self.y_val=np.array(y_val)
        self.x_test=np.array(x_test)
        self.y_test=np.array(y_test)
        return self.x_train, self.x_val,self.x_test, self.y_train, self.y_val,self.y_test, self.labels    


if __name__ == "__main__":
    df = pd.read_csv("/home/shefali/Documents/Declaration_of_consent/Data_recording_codes.csv")
    psuedo_codes = list(df['Anonymous code'])
    #print(psuedo_codes)
    repo=DataRepository(psuedo_codes)
    BB_X_train1, BB_X_val1, BB_X_test1, BB_Y_train1, BB_Y_val1, BB_Y_test1, BB_Labels1 = repo.getForTraining()
    print('X_train values',BB_X_train1.shape)
    print('Y_val values',(BB_Y_val1.shape))
    print('X_test values', (BB_X_test1.shape))
    print(type(BB_X_train1))
    np.save('BB_X_train_new.npy',BB_X_train1 )
    np.save('BB_X_test_new.npy',BB_X_test1)
    np.save('BB_X_val_new.npy',BB_X_val1)
    np.save('BB_Y_test_new.npy',BB_Y_test1)
    np.save('BB_Y_train_new.npy',BB_Y_train1)
    np.save('BB_Y_val_new.npy',BB_Y_val1)
    np.save('BB_Labels_new.npy',BB_Labels1)
