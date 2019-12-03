import glob
import pandas as pd
from pylab import *
%matplotlib inline
def loading_data(folder):
    X = []
    y = []
    for file in glob.glob(folder+"/*"):
        s = pd.read_csv(file,names=["x","y","z","state"],sep=" ")
        X.append(np.array([np.array(s.x),np.array(s.y)]))
        y.append(np.array(s.state,dtype=np.int))
    return X,y


Xt,y = loading_data("../data/wetransfer-7200d4/")

def generate_data(X,y,size):
    return np.array(X[:size]),np.array(y[:size]),np.array(X[size:]),np.array(y[size:])
Xtrain,ytrain,Xtest,ytest = generate_data(Xt,y,3500)


from keras.models import Sequential, Model
from keras.layers import Dense, Flatten, AveragePooling1D, TimeDistributed, Dropout, Input, Bidirectional
from keras.layers import LSTM
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=5, padding='same',
                 activation='relu', input_shape=(2400, 2)))
model.add(LSTM(10))
model.add(TimeDistributed(Dense(1, activation='sigmoid')))

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam')

model.fit(Xtrain, ytrain, epochs=40,validation_data=[Xtest,ytest])
