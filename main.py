from src.Seeker import Seeker
from src.IMGAnalyzer import IMGAnalyzer
from src.DSBuilder import DSBuilder
from keras.models import Sequential 
from keras.layers import Dense, Activation 
from keras.layers import Conv2D, Flatten
from keras.layers import MaxPooling2D, Dropout
from keras import optimizers

import numpy as np

def train(data):
    n = len(data)

    # 80% train 20% validation
    X_train = np.asarray([y for (_,y,_) in data[:int(n * 0.8)]])
    Y_train  = np.asarray([z for (_,_,z) in data[:int(n * 0.8)]])
    
    X_test = np.asarray([y for (_,y,_) in data[int(n * 0.8):]])
    Y_test  = np.asarray([z for (_,_,z) in data[int(n * 0.8):]])
    
    print(X_train.shape)
    print(Y_train.shape)
    print(X_test.shape)
    print(Y_test.shape)
    
    
    
    # create keras model 
    model = Sequential()

    model.add(Conv2D(16, (3, 3), activation='relu', input_shape=(480, 640, 3)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    

    model.add(Flatten())

    model.add(Dense(128, activation='relu'))   
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(7, activation="softmax"))
    
    sgd = optimizers.SGD(lr=0.005)
    model.compile(loss="categorical_crossentropy", 
    metrics=['accuracy'], optimizer=sgd)
    
    print(model.summary())
    model.fit(X_train, Y_train, validation_data=(X_test, Y_test), batch_size=64, epochs=40)

def main():
        
    

    dataset_dir = DSBuilder()
    dataset_dir.seek()

    train(dataset_dir.dataset)

if __name__ == '__main__':
    main()