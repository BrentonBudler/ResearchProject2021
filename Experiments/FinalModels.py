import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn import metrics
from sklearn import svm 

import tensorflow as tf
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D, Flatten, SimpleRNN, LSTM, TimeDistributed
from tensorflow.keras.callbacks import EarlyStopping 
from tensorflow.keras.utils import plot_model
from keras.layers import CuDNNLSTM

from timeit import timeit

tf.random.set_seed(42)
np.random.seed(42)

class SVM:
    def __init__(self, X, y, split, ae= False, ae_dim =0, ae_train_perc = 0.25):
        self.split = split 
        self.ae = ae 
        self.ae_dim = ae_dim
        self.ae_train_perc = ae_train_perc
        
        # For CSE-CIC-IDS2018 
        if self.split:
            self.X = X 
            self.y = y
        # For NSL-KDD (Already split by providers)
        else: 
            self.X_train, self.X_test = X
            self.y_train, self.y_test = y 
            
    def preprocess_data(self):
        
        if self.split:
            # Split data into train and test split 
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, stratify = self.y)
        else:
            X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
        
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.transform(X_test)
        
        if self.ae: 
            _ , X_val = train_test_split(X_train, test_size=self.ae_train_perc, random_state=42)
            
            encoder = Sequential()
            encoder.add(Dense(self.ae_dim, activation='relu',input_shape=[X_val.shape[1]]))

            decoder = Sequential()
            decoder.add(Dense(X_val.shape[1], activation='relu', input_shape=[self.ae_dim]))

            autoencoder = Sequential([encoder,decoder])
            autoencoder.compile(loss='mse',optimizer = 'adam')

            early_stop = EarlyStopping(monitor='loss', mode = 'auto', 
                                       patience = 5 , verbose =0, restore_best_weights=True)

            autoencoder.fit(X_val, X_val, epochs=40, batch_size=256,verbose=0, callbacks=[early_stop])

            X_train = encoder.predict(X_train)
            X_test = encoder.predict(X_test)
        
        return X_train, X_test, y_train, y_test
        
    def evaluate(self, runs):
        results = pd.DataFrame(columns=["acc","prec","rec","f1","far"])
        reports = []    
        
        for i in range(runs):
            X_train, X_test, y_train, y_test = self.preprocess_data()

            model = svm.SVC()
            model.fit(X_train,y_train)

            y_pred = model.predict(X_test)

            acc = metrics.accuracy_score(y_test,y_pred)
            prec = metrics.precision_score(y_test,y_pred,average='weighted')
            rec = metrics.recall_score(y_test,y_pred,average='weighted')
            f1 = metrics.f1_score(y_test,y_pred,average='weighted')
            
            confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
            FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
            FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
            TP = np.diag(confusion_matrix)
            TN = confusion_matrix.sum() - (FP + FN + TP)
            FAR = FP/(FP+TN)
            
            model_results = {'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1,'far':FAR.mean()}
            results = results.append(model_results, ignore_index= True)
            print(f"Model {i} Acc:", acc)
            
            report = pd.DataFrame(metrics.classification_report(y_test,y_pred,output_dict=True)).transpose()
            reports.append(report)
            
        self.acc = results.mean()[0], results.std()[0]
        self.prec = results.mean()[1], results.std()[1]
        self.rec = results.mean()[2], results.std()[2]
        self.f1 = results.mean()[3], results.std()[3]
        self.far = results.mean()[4], results.std()[4]
        self.report = pd.concat(reports).groupby(level=0).mean()  
        

class DNN:
    def __init__(self, X, y, split, ae= False, ae_dim =0 , ae_train_perc = 0.25):
        self.split = split 
        self.ae = ae 
        self.ae_dim = ae_dim
        self.ae_train_perc = ae_train_perc
        
        # For CSE-CIC-IDS2018 
        if self.split:
            self.X = X 
            self.y = y
        # For NSL-KDD (Already split by providers)
        else: 
            self.X_train, self.X_test = X
            self.y_train, self.y_test = y 
            
        
    def preprocess_data(self):
        if self.split:
            # Split data into train and test split 
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, stratify = self.y)
        else:
            X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test
            
        # Scale Data 
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.transform(X_test)
        
        # Encode Labels
        y_train = pd.get_dummies(y_train).values
        y_test = pd.get_dummies(y_test).values
        
        # Use Auto Encoder to perform dimensionality reduction 
        if self.ae: 
            _ , X_val = train_test_split(X_train, test_size=self.ae_train_perc, random_state=42)
            
            encoder = Sequential()
            encoder.add(Dense(self.ae_dim, activation='relu',input_shape=[X_val.shape[1]]))

            decoder = Sequential()
            decoder.add(Dense(X_val.shape[1], activation='relu', input_shape=[self.ae_dim]))

            autoencoder = Sequential([encoder,decoder])
            autoencoder.compile(loss='mse',optimizer = 'adam')

            early_stop = EarlyStopping(monitor='loss', mode = 'auto', 
                                       patience = 5 , verbose =0, restore_best_weights=True)

            autoencoder.fit(X_val, X_val, epochs=40, batch_size=256,verbose=0, callbacks=[early_stop])

            X_train = encoder.predict(X_train)
            X_test = encoder.predict(X_test)
        
        return X_train, y_train, X_test, y_test
        
    def build_model(self, hidden_layers):
        
        # Split data into training and testing set, scale it and encode labels 
        X_train, y_train, X_test, y_test = self.preprocess_data()
        
        # Further divide Data into training and validation data
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)
        
        model = Sequential()
        
        for layer in hidden_layers:
            model.add(Dense(layer, activation='relu'))
            model.add(BatchNormalization())
            model.add(Dropout(0.2))

        model.add(Dense(y_train.shape[1], activation = 'softmax'))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        
        early_stop = EarlyStopping(monitor='val_loss', mode = 'auto', 
                                   patience = 5 , verbose =0, restore_best_weights=True)
        
        model.fit(x = X_train, y = y_train, epochs = 200, batch_size = 256, verbose = 0,
                   validation_data = (X_val,y_val),callbacks = [early_stop])
        
        return model, X_test, y_test
    
    def evaluate_model(self, runs, hidden_layers, time=False):
        results = pd.DataFrame(columns=["acc","prec","rec","f1","far","Train Time","Test Time"])
        reports = []
        
        # Train and evaluate model 
        for i in range(runs):
            print(f"Training Model {i} ...")
            
            if time:
                training_time = timeit(lambda: self.build_model(hidden_layers), number = 1)
            else:
                training_time= np.nan
                
            model, X_test, y_test = self.build_model(hidden_layers)
            
            print(f"Evaluating Model {i} ...")
            if time:
                testing_time = timeit(lambda: model.predict(X_test), number = 1)
            else:
                testing_time = np.nan
      
            y_pred = np.argmax(model.predict(X_test), axis=-1)
            y_org = np.argmax(y_test,axis=1)
            
            acc = metrics.accuracy_score(y_org,y_pred)
            prec = metrics.precision_score(y_org,y_pred,average='weighted')
            rec = metrics.recall_score(y_org,y_pred,average='weighted')
            f1 = metrics.f1_score(y_org,y_pred,average='weighted')
            
            confusion_matrix = metrics.confusion_matrix(y_org,y_pred)
            FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
            FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
            TP = np.diag(confusion_matrix)
            TN = confusion_matrix.sum() - (FP + FN + TP)
            FAR = FP/(FP+TN)
            
            model_results = {'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1,'far':FAR.mean(),
                             'Train Time':training_time,'Test Time':testing_time}
            results = results.append(model_results, ignore_index= True)
            print(f"Model {i} Acc:", acc)
            
            # Creation of a classification report which is useful for examining per class metrics
            report = pd.DataFrame(metrics.classification_report(y_org,y_pred,output_dict=True)).transpose()
            reports.append(report)
        
        self.acc = results.mean()[0], results.std()[0]
        self.prec = results.mean()[1], results.std()[1]
        self.rec = results.mean()[2], results.std()[2]
        self.f1 = results.mean()[3], results.std()[3]
        self.far = results.mean()[4], results.std()[4]
        self.train_time = results.mean()[5], results.std()[5]
        self.test_time = results.mean()[6], results.std()[6]
        self.report = pd.concat(reports).groupby(level=0).mean()
        
class CNN:
    def __init__(self, X, y, split, image_dim, ae= False, ae_dim =0, ae_train_perc = 0.25):
        self.split = split 
        self.ae = ae 
        self.ae_dim = ae_dim
        self.image_dim = image_dim
        self.ae_train_perc = ae_train_perc
        
        # For CSE-CIC-IDS2018 
        if self.split:
            # Add two extra features for 2D image inputs 
            a = np.zeros((len(X),1))
            b = np.zeros((len(X),1))
            self.X = np.hstack((X,a,b))
            self.y = y
        # For NSL-KDD (Already split by providers)
        else: 
            self.X_train, self.X_test = X
            self.y_train, self.y_test = y 
            
    def preprocess_data(self):
        
        if self.split:
            # Split data into train and test split 
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, stratify = self.y)
        else:
            X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        # Scale Data 
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.transform(X_test)

        # Encode Labels
        y_train = pd.get_dummies(y_train).values
        y_test = pd.get_dummies(y_test).values

        # Use Auto Encoder to perform dimensionality reduction 
        if self.ae: 
            _ , X_val = train_test_split(X_train, test_size=self.ae_train_perc, random_state=42)
            
            encoder = Sequential()
            encoder.add(Dense(self.ae_dim, activation='relu',input_shape=[X_val.shape[1]]))

            decoder = Sequential()
            decoder.add(Dense(X_val.shape[1], activation='relu', input_shape=[self.ae_dim]))

            autoencoder = Sequential([encoder,decoder])
            autoencoder.compile(loss='mse',optimizer = 'adam')

            early_stop = EarlyStopping(monitor='loss', mode = 'auto', 
                                       patience = 5 , verbose =0, restore_best_weights=True)

            autoencoder.fit(X_val, X_val, epochs=40, batch_size=256,verbose=0, callbacks=[early_stop])

            X_train = encoder.predict(X_train)
            X_test = encoder.predict(X_test)

        # Reshape the data into 2D input arrays
        X_train = X_train.reshape(len(X_train), self.image_dim, self.image_dim,1)
        X_test = X_test.reshape(len(X_test), self.image_dim, self.image_dim,1) 

        return X_train, y_train, X_test, y_test
        
    def display_class_image(self, classes):
        X_train, y_train, X_test, y_test = self.preprocess_data()
        y_train_ser = pd.Series(np.argmax(y_train,axis=1))
        fig, axs = plt.subplots(nrows =1 , ncols = (len(classes)), figsize= (15,6))
        for c in classes.keys():
            X_image = X_train[y_train_ser[y_train_ser==c].index[0]]
            axs[c].imshow(X_image, cmap='Greys_r')
            axs[c].set_title(classes[c])
            axs[c].axis('off')
        plt.show()
        
    def build_model(self):
        
        # Split data into training and testing set, scale it and encode labels 
        X_train, y_train, X_test, y_test = self.preprocess_data()
    
        # Further divide Data into training and validation data
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, stratify= y_train)

        model = Sequential()

        model.add(Conv2D(filters=32, kernel_size=(2,2), input_shape=(self.image_dim,self.image_dim,1), padding = 'same', activation='relu'))
        model.add(BatchNormalization())
        
        model.add(MaxPool2D(pool_size=(2, 2)))

        
        model.add(Conv2D(filters=16, kernel_size=(2,2), padding = 'same', activation='relu'))
        model.add(MaxPool2D(pool_size=(2, 2)))
        
        model.add(Flatten())
        
        model.add(Dense(64,activation='relu'))
        model.add(Dropout(0.2))
        
        model.add(Dense(y_train.shape[1],activation = "softmax"))

        model.compile(loss="categorical_crossentropy", optimizer = 'adam')
        
        early_stop = EarlyStopping(monitor='val_loss', mode = 'auto', 
                                   patience = 5 , verbose = 0, restore_best_weights=True)
        
        model.fit(x = X_train, y = y_train, epochs = 100, batch_size = 256, verbose = 0, 
                  callbacks = [early_stop], validation_data = (X_val,y_val))
        
        return model, X_test, y_test
        
 
    def evaluate_model(self, runs, time=False):
        results = pd.DataFrame(columns=["acc","prec","rec","f1","far","Train Time","Test Time"])
        reports = []
        
        # Train and evaluate model 
        for i in range(runs):
            
            print(f"Training Model {i} ...")
            
            if time:
                training_time = timeit(lambda: self.build_model(), number = 1)
            else:   
                training_time= np.nan 
            model, X_test, y_test = self.build_model()
            
            print(f"Evaluating Model {i} ...")
            
            if time:
                  testing_time = timeit(lambda: model.predict(X_test), number = 1)
            else:        
                testing_time = np.nan
                
            y_pred = np.argmax(model.predict(X_test), axis=-1)
            y_org = np.argmax(y_test,axis=1)
            
            acc = metrics.accuracy_score(y_org,y_pred)
            prec = metrics.precision_score(y_org,y_pred,average='weighted')
            rec = metrics.recall_score(y_org,y_pred,average='weighted')
            f1 = metrics.f1_score(y_org,y_pred,average='weighted')
            
            confusion_matrix = metrics.confusion_matrix(y_org,y_pred)
            FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
            FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
            TP = np.diag(confusion_matrix)
            TN = confusion_matrix.sum() - (FP + FN + TP)
            FAR = FP/(FP+TN)
            
            model_results = {'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1,'far':FAR.mean(),
                             'Train Time':training_time,'Test Time':testing_time}
            results = results.append(model_results, ignore_index= True)
            print(f"Model {i} Acc:", acc)
            
            # Creation of a classification report which is useful for examining per class metrics
            report = pd.DataFrame(metrics.classification_report(y_org,y_pred,output_dict=True)).transpose()
            reports.append(report)
        
        self.acc = results.mean()[0], results.std()[0]
        self.prec = results.mean()[1], results.std()[1]
        self.rec = results.mean()[2], results.std()[2]
        self.f1 = results.mean()[3], results.std()[3]
        self.far = results.mean()[4], results.std()[4]
        self.train_time = results.mean()[5], results.std()[5]
        self.test_time = results.mean()[6], results.std()[6]
        self.report = pd.concat(reports).groupby(level=0).mean()
        
class RNN_LSTM:
    def __init__(self, X, y, split, num_of_classes, ae= False, ae_dim =0, ae_train_perc = 0.25):
        self.split = split 
        self.ae = ae 
        self.ae_dim = ae_dim
        self.num_of_classes = num_of_classes
        self.ae_train_perc = ae_train_perc
        
        # For CSE-CIC-IDS2018 
        if self.split:
            # Add two extra features for 2D image inputs 
            a = np.zeros((len(X),1))
            b = np.zeros((len(X),1))
            self.X = np.hstack((X,a,b))
            self.y = y
        # For NSL-KDD (Already split by providers)
        else: 
            self.X_train, self.X_test = X
            self.y_train, self.y_test = y 
            
    def preprocess_data(self):
        
        if self.split:
            # Split data into train and test split 
            X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.3, stratify = self.y)
        else:
            X_train, X_test, y_train, y_test = self.X_train, self.X_test, self.y_train, self.y_test

        # Scale Data 
        min_max_scaler = MinMaxScaler()
        X_train = min_max_scaler.fit_transform(X_train)
        X_test = min_max_scaler.transform(X_test)

        # Use Auto Encoder to perform dimensionality reduction 
        if self.ae: 
            _ , X_val = train_test_split(X_train, test_size=self.ae_train_perc, random_state=42)
            
            encoder = Sequential()
            encoder.add(Dense(self.ae_dim, activation='relu',input_shape=[X_val.shape[1]]))

            decoder = Sequential()
            decoder.add(Dense(X_val.shape[1], activation='relu', input_shape=[self.ae_dim]))

            autoencoder = Sequential([encoder,decoder])
            autoencoder.compile(loss='mse',optimizer = 'adam')

            early_stop = EarlyStopping(monitor='loss', mode = 'auto', 
                                       patience = 5 , verbose =0, restore_best_weights=True)

            autoencoder.fit(X_val, X_val, epochs=40, batch_size=256,verbose=0, callbacks=[early_stop])

            X_train = encoder.predict(X_train)
            X_test = encoder.predict(X_test)

        # Reshape the data into 3D input arrays which include a time dimension as input for LSTM
        X_train = X_train.reshape(len(X_train), 1, X_train.shape[1])
        X_test = X_test.reshape(len(X_test), 1, X_test.shape[1]) 

        return X_train, y_train, X_test, y_test
      
        
    def build_model(self):
        # Split data into training and testing set, scale it and encode labels 
        X_train, y_train, X_test, y_test = self.preprocess_data()
        
        # Further divide Data into training and validation data
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42, stratify= y_train)
        
        input_dim = X_train.shape[1]
        model = Sequential()

        model.add(LSTM(64, input_shape = (1,X_train.shape[2]), return_sequences=True, unroll=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))


        model.add(LSTM(32,unroll=True))
        model.add(BatchNormalization())
        model.add(Dropout(0.2))


        model.add(Dense(self.num_of_classes, activation ='softmax'))
        model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

        early_stop = EarlyStopping(monitor='val_loss', mode = 'auto', 
                           patience = 5 , verbose =1, restore_best_weights=True)

        model.fit(x = X_train, y = y_train, epochs = 100, batch_size = 256, verbose = 0,
                  callbacks = [early_stop], validation_data = (X_val,y_val))
        
        return model, X_test, y_test
    
    def evaluate_model(self, runs,time=False):
        results = pd.DataFrame(columns=["acc","prec","rec","f1","far","Train Time","Test Time"])
        reports = []
        
        self.preprocess_data()
        
        for i in range(runs):
            print(f"Training Model {i} ...")
            if time:
                training_time = timeit(lambda: self.build_model(), number = 1)
            else:
                training_time= np.nan
                
            model, X_test, y_test = self.build_model()
            print(f"Evaluating Model {i} ...")
            
            if time:
                testing_time = timeit(lambda: model.predict(X_test), number = 1)
            else:
                testing_time=1
                
            y_pred = np.argmax(model.predict(X_test), axis=-1)
            
            acc = metrics.accuracy_score(y_test,y_pred)
            prec = metrics.precision_score(y_test,y_pred,average='weighted')
            rec = metrics.recall_score(y_test,y_pred,average='weighted')
            f1 = metrics.f1_score(y_test,y_pred,average='weighted')
            
            confusion_matrix = metrics.confusion_matrix(y_test,y_pred)
            FP = confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)  
            FN = confusion_matrix.sum(axis=1) - np.diag(confusion_matrix)
            TP = np.diag(confusion_matrix)
            TN = confusion_matrix.sum() - (FP + FN + TP)
            FAR = FP/(FP+TN)
            
            model_results = {'acc':acc, 'prec':prec, 'rec':rec, 'f1':f1,'far':FAR.mean(),
                             'Train Time':training_time,'Test Time':testing_time}
            results = results.append(model_results, ignore_index= True)
            print(f"Model {i} Acc:", acc)
            
            # Creation of a classification report which is useful for examining per class metrics
            report = pd.DataFrame(metrics.classification_report(y_test,y_pred,output_dict=True)).transpose()
            reports.append(report)
        
        self.acc = results.mean()[0], results.std()[0]
        self.prec = results.mean()[1], results.std()[1]
        self.rec = results.mean()[2], results.std()[2]
        self.f1 = results.mean()[3], results.std()[3]
        self.far = results.mean()[4], results.std()[4]
        self.train_time = results.mean()[5], results.std()[5]
        self.test_time = results.mean()[6], results.std()[6]
        self.report = pd.concat(reports).groupby(level=0).mean()