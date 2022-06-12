import tensorflow as tf
import keras
from keras.models import Model
from keras.layers import *
from keras.optimizers import *
from losses.huber_loss import huber_loss_mean
from model.scse import channel_spatial_squeeze_excite

def QFunction(inputs_):
    o = Conv2D(32, (3, 3), padding='same', kernel_initializer='random_uniform')(inputs_)
    o = channel_spatial_squeeze_excite(o)
    o = MaxPooling2D((2, 2))(o)
    o = Conv2D(64, (3, 3), padding='same', kernel_initializer='random_uniform')(o)
    o = channel_spatial_squeeze_excite(o)
    o = MaxPooling2D((2, 2))(o)
    o = Conv2D(64, (3, 3), padding='same', kernel_initializer='random_uniform')(o)
    o = channel_spatial_squeeze_excite(o)
    o = Flatten()(o)
    o = Dense(2, activation='linear')(o)
    model = Model(inputs=inputs_, outputs=o)
    model.compile(optimizer=Adam(lr=0.001), loss=huber_loss_mean, metrics=['acc'])
    return model