from src.Seeker import Seeker
from src.IMGAnalyzer import IMGAnalyzer
from src.DSBuilder import DSBuilder
from keras.models import Sequential 
from keras.layers import Dense, Activation 
from keras.layers import Conv2D, Flatten
from keras.layers import MaxPooling2D, Dropout
from keras import optimizers
import sys
import numpy as np


def generator(batch_size,from_list_x,from_list_y):

    assert len(from_list_x) == len(from_list_y)
    total_size = len(from_list_x)

    while True: #keras generators should be infinite
        for i in range(0,total_size,batch_size):
            yield np.array(from_list_x[i:i+batch_size]), np.array(from_list_y[i:i+batch_size])

def train(data):
    n = len(data)
   
    # 80% train 20% validation
    X_train = [y for (_,y,_) in data[:int(n * 0.8)]]
    Y_train  =[z for (_,_,z) in data[:int(n * 0.8)]]
    
    X_test = [y for (_,y,_) in data[int(n * 0.8):]]
    Y_test  = [z for (_,_,z) in data[int(n * 0.8):]]
    

    # create keras model 
    model = Sequential()

    model.add(Conv2D(64, (3, 3), activation='relu', input_shape=(218, 218, 3)))
    
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    
    model.add(Conv2D(64, (3,3), activation="relu"))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())

    model.add(Dense(7, activation="relu"))
    model.add(Dropout(0.25))
    
    opt = optimizers.SGD(lr=0.1)
    model.compile(loss="categorical_crossentropy", 
    metrics=['accuracy'], optimizer=opt)


    
    model.fit_generator(generator(64,X_train,Y_train), 
    steps_per_epoch=len(X_train)//64, 
    epochs=20)
    # model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=64, epochs=120)

def main():
        
    
    
    dataset_dir = DSBuilder()
    dataset_dir.seek()

    train(dataset_dir.dataset)

if __name__ == '__main__':
    main()