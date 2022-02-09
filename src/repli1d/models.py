import numpy as np
import tensorflow as tf
from keras import backend as K
from keras import metrics
from keras.models import Model, load_model
from tensorflow.keras.callbacks import (EarlyStopping, History,
                                        ModelCheckpoint, ReduceLROnPlateau)
from tensorflow.keras.layers import (Activation, Add, Conv1D, Conv2D, Dense,
                                     Dropout, Flatten, Input, MaxPooling1D,
                                     MaxPooling2D)

def jm_cnn_model(X_train, targets, nfilters, kernel_length,
                 loss="binary_crossentropy"):
    """The model that Jean Michel has implemented, and trained.
    """
    dropout = 0.2
    dropout = 0.01
    K.set_image_data_format('channels_last')

    multi_layer_keras_model = tf.keras.Sequential()
    multi_layer_keras_model.add(Input(X_train.shape[1:]))
    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), activation='relu'))
    multi_layer_keras_model.add(Dropout(dropout))

    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), activation='relu'))
    multi_layer_keras_model.add(Dropout(dropout))

    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), activation='relu'))
    multi_layer_keras_model.add(MaxPooling2D(pool_size=(1, 2)))
    multi_layer_keras_model.add(Dropout(dropout))

    multi_layer_keras_model.add(Flatten())
    multi_layer_keras_model.add(Dense(1, activation='sigmoid'))
    # multi_layer_keras_model.compile(optimizer='adadelta',  # 'adam'
    #                                loss='mean_squared_logarithmic_error')
    ada = tf.keras.optimizers.Adadelta(
    learning_rate=1, rho=0.95, epsilon=1e-07, name="Adadelta")
    multi_layer_keras_model.compile(optimizer=ada,  # 'adam'
                                    loss=loss,
                                    metrics=[metrics.MSE])
    multi_layer_keras_model.summary()
    return multi_layer_keras_model

def jm_cnn_model_beta(X_train, targets, nfilters, kernel_length,
                 loss="mse"):
    """Some slight modifications on the model that Jean Michel has implemented,
    and trained. this nn can be a costumized to compare the effect of optimizers,
    dropout, and activation functions.
    """
    dropout = 0.2
    dropout = 0.01
    K.set_image_data_format('channels_last')
    multi_layer_keras_model = tf.keras.Sequential()
    multi_layer_keras_model.add(Input(X_train.shape[1:]))
    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), activation='relu'))
    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), activation='relu'))
    multi_layer_keras_model.add(Conv2D(filters=nfilters, kernel_size=(
        1, kernel_length), activation='relu'))
    multi_layer_keras_model.add(MaxPooling2D(pool_size=(1, 2)))
    multi_layer_keras_model.add(Flatten())
    multi_layer_keras_model.add(Dense(1, activation='linear'))
    multi_layer_keras_model.compile(optimizer='adam',
                                    loss=loss,
                                    metrics=[metrics.MSE])
    multi_layer_keras_model.summary()
    return multi_layer_keras_model

def residual_block(x, f):
    """Residual blocks for a resnet.
    """
    x = Conv1D(f, 1, strides=1, padding='same', data_format='channels_last')(x)
    x = Activation('relu')(x)
    x_shortcut = x
    out = Conv1D(f, 3, strides=1, padding="same", data_format='channels_last')(x)
    out = Activation('relu')(out)
    out = Conv1D(f, 3, strides=1, padding="same", data_format='channels_last')(out)
    out = Activation('relu')(out)
    if out.shape[1:] == x_shortcut.shape[1:]:
      x = Add()([out, x_shortcut])
    else:
      raise Exception('Skip Connection Failure!')
    x_shortcut = x
    out = Conv1D(f, 3, strides=1, padding="same", data_format='channels_last')(x)
    out = Activation('relu')(out)
    out = Conv1D(f, 3, strides = 1, padding = "same", data_format='channels_last')(out)
    out = Activation('relu')(out)
    if out.shape[1:] == x_shortcut.shape[1:]:
      x = Add()([out, x_shortcut])
    else:
      raise Exception('Skip Connection Failure!')
    x = MaxPooling1D(pool_size=2, strides=None, padding='valid', data_format='channels_last')(x)
    return x

def resnet(input_shape, classes):
    """Residual Neural Network.
    to call the model please use this sample:
    model = resnet([number of datapoints in each sequence, number of sequences], number of outputs)
    for using with window stacking:
    model = resnet([number of datapoints in each windnow, number of sequences], number of outputs)
    """
    x_input = Input(input_shape)
    x = x_input
    num_filters = 8
    x = residual_block(x, num_filters)
    x = residual_block(x, 2*num_filters)
    x = residual_block(x, 4*num_filters)
    x = residual_block(x, 8*num_filters)
    x = residual_block(x, 16*num_filters)
    x = Flatten()(x)
    x = Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = Dense(128, activation="relu", kernel_initializer="he_normal")(x)
    x = Dense(classes , activation='linear', kernel_initializer = "he_normal")(x)
    model = Model(inputs = x_input, outputs = x)
    return model
