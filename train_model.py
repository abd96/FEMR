from src.IMGAnalyzer import IMGAnalyzer
from src.DSBuilder import DSBuilder
from keras.models import Sequential 
from keras.layers import Dense, Activation
from keras.layers import Conv2D, Flatten
from keras.layers import MaxPooling2D, Dropout, BatchNormalization
from keras import optimizers
from keras.utils import np_utils
from src.FeatureExtractor import  FeatureExtractor
import sys, os
import numpy as np
import pandas as pd
import random 
import traceback 


def generator(batch_size,from_list_x,from_list_y):

    assert len(from_list_x) == len(from_list_y)
    total_size = len(from_list_x)

    while True: #keras generators should be infinite
        for i in range(0,total_size,batch_size):
            yield np.array(from_list_x[i:i+batch_size]), np.array(from_list_y[i:i+batch_size])

def train(path):
    
    dataframe = pd.read_csv(path)
    X_train, Y_train, X_test, Y_test = [],[],[],[]
    # Reading data from 
    for i, row in dataframe.iterrows():
        pixels  =  row['pixels'].split(' ')
        try:
            # train data
            if 'Training' == row['Usage']:
                X_train.append(np.array(pixels, 'float32'))
                Y_train.append(row['emotion'])
            # test data 
            elif 'PublicTest' ==  row['Usage']:
                X_test.append(np.array(pixels, 'float32'))
                
                Y_test.append(row["emotion"])
        except:
            raise Exception(f"Error occured while reading input data from  {path}")
            traceback.print_exc()

    # Convert data from list to numpy arrays
    X_train = np.array(X_train, 'float32') 
    Y_train = np.array(Y_train, 'float32') 
    X_train = np.array(X_train, 'float32') 
    Y_train = np.array(Y_train, 'float32') 

    # Normalize pixels
    X_train - np.mean(X_train, axis= 0)
    X_train /= np.std(X_train, axis=0)

    X_test - np.mean(X_test, axis= 0)
    X_test /= np.std(X_test, axis=0)

    width, height = 48,48
    
    # reshape to image shapes of 28 * 28 
    X_train = X_train.reshape(X_train.shape[0], width, height, 1)  # 1 for channel one grey scaled data 
    X_test  = X_test.reshape(X_test.shape[0], width, height, 1)  # 1 for channel one grey scaled data 

    # model design
    num_features = 64 
    num_labels = 7
    batch_size = 64
    epochs = 100


    # categorize targets 
    Y_train = np_utils.to_categorical(Y_train , num_classes = num_labels)
    Y_test = np_utils.to_categorical(Y_test , num_classes = num_labels)   
   
    
    # design keras model 
    model = Sequential()
    
    model.add(Conv2D(num_features, (3, 3), activation='relu', input_shape=(width, height, 1)))
    model.add(Conv2D(num_features, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(num_features,(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(num_features, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))
    model.add(Dropout(0.5))

    model.add(Conv2D(2*num_features,(3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(2*num_features, (3, 3), activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D(pool_size=(2,2), strides = (2,2)))

    model.add(Flatten())
    model.add(Dense(2*2*2*2*num_features, activation="softmax"))
    model.add(Dropout(0.2))
    model.add(Dense(num_labels, activation="softmax"))
    
    
    opt = optimizers.SGD(lr=0.001)
    model.compile(loss="categorical_crossentropy", 
    metrics=['accuracy'], optimizer=optimizers.Adam())


    print(model.summary())
    # model.fit_generator(generator(32,X_train,Y_train), 
    # steps_per_epoch=len(X_train)//32,
    # epochs=50, 
    # verbose=1,
    # shuffle = True)
    
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test),batch_size=batch_size, 
    epochs=epochs,
    verbose=1,
    shuffle=True)

    model_json = model.to_json()
    with open("model.json", "w") as file:
        file.write(model.json)
    model.save_weights("model_weights.h5") 


def main():

    path_to_trainset =  os.path.join(os.getcwd(), 'FEXDB', 'fer2013', 'fer2013', 'fer2013.csv')
    train(path_to_trainset) 

if __name__ == '__main__':
    main()