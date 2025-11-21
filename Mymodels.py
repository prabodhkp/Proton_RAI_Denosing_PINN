# -*- coding: utf-8 -*-
"""
Created on Thu Dec  9 17:14:43 2021

@author: TRUE
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 30 19:53:36 2021

@author: TRUE
"""

# import scipy.io as sio
import numpy as np 
import keras as KE
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Conv3D, MaxPooling3D, Dense, Input, Reshape, Concatenate, BatchNormalization, Dropout, Activation, UpSampling3D, Lambda
import tensorflow.keras.optimizers as KO
from keras.callbacks import ModelCheckpoint, LearningRateScheduler, Callback
from keras.initializers import RandomNormal
from keras import regularizers
import keras.backend as K



def unet_3d(input_size = (96,96,112)):
    dropout_rate = 0.5
    filter_base = 32

    input_uCBCT = Input(input_size + (1,), name='main_input_uCBCT')

    conv1_uCBCT = Conv3D(filter_base, 3, padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                   bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                   kernel_regularizer=regularizers.l2(1e-4), name='uCBCT_conv1_1')(input_uCBCT)
    conv1_uCBCT = BatchNormalization(name='uCBCT_BN1_1')(conv1_uCBCT)
    conv1_uCBCT = Lambda(lambda x: K.relu(x))(conv1_uCBCT)
    conv1_uCBCT = Conv3D(filter_base, 3, padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                   bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                   kernel_regularizer=regularizers.l2(1e-4), name='uCBCT_conv1_2')(conv1_uCBCT)
    conv1_uCBCT = BatchNormalization(name='uCBCT_BN1_2')(conv1_uCBCT)
    conv1_uCBCT = Lambda(lambda x: K.relu(x))(conv1_uCBCT)
    conv1_uCBCT = Conv3D(filter_base, 3, padding='same',
                   kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                   bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                   kernel_regularizer=regularizers.l2(1e-4), name='uCBCT_conv1_3')(conv1_uCBCT)
    conv1_uCBCT = BatchNormalization(name='uCBCT_BN1_3')(conv1_uCBCT)
    conv1_uCBCT = Lambda(lambda x: K.relu(x))(conv1_uCBCT)
    pool1_uCBCT = MaxPooling3D(pool_size=(2, 2, 2))(conv1_uCBCT)
    conv2_uCBCT = Conv3D(2 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4), name='uCBCT_conv2_1')(pool1_uCBCT)
    conv2_uCBCT = BatchNormalization(name='uCBCT_BN2_1')(conv2_uCBCT)
    conv2_uCBCT = Lambda(lambda x: K.relu(x))(conv2_uCBCT)
    conv2_uCBCT = Conv3D(2 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4), name='uCBCT_conv2_2')(conv2_uCBCT)
    conv2_uCBCT = BatchNormalization(name='uCBCT_BN2_2')(conv2_uCBCT)
    conv2_uCBCT = Lambda(lambda x: K.relu(x))(conv2_uCBCT)
    conv2_uCBCT = Conv3D(2 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4), name='uCBCT_conv2_3')(conv2_uCBCT)
    conv2_uCBCT = BatchNormalization(name='uCBCT_BN2_3')(conv2_uCBCT)
    conv2_uCBCT = Lambda(lambda x: K.relu(x))(conv2_uCBCT)
    pool2_uCBCT = MaxPooling3D(pool_size=(2, 2, 2))(conv2_uCBCT)
    conv3_uCBCT = Conv3D(4 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4), name='uCBCT_conv3_1')(pool2_uCBCT)
    conv3_uCBCT = BatchNormalization(name='uCBCT_BN3_1')(conv3_uCBCT)
    conv3_uCBCT = Lambda(lambda x: K.relu(x))(conv3_uCBCT)
    conv3_uCBCT = Conv3D(4 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4), name='uCBCT_conv3_2')(conv3_uCBCT)
    conv3_uCBCT = BatchNormalization(name='uCBCT_BN3_2')(conv3_uCBCT)
    conv3_uCBCT = Lambda(lambda x: K.relu(x))(conv3_uCBCT)
    conv3_uCBCT = Conv3D(4 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4), name='uCBCT_conv3_3')(conv3_uCBCT)
    conv3_uCBCT = BatchNormalization(name='uCBCT_BN3_3')(conv3_uCBCT)
    conv3_uCBCT = Lambda(lambda x: K.relu(x))(conv3_uCBCT)
    pool3_uCBCT = MaxPooling3D(pool_size=(2, 2, 2))(conv3_uCBCT)
    conv4_uCBCT = Conv3D(8 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4), name='uCBCT_conv4_1')(pool3_uCBCT)
    conv4_uCBCT = BatchNormalization(name='uCBCT_BN4_1')(conv4_uCBCT)
    conv4_uCBCT = Lambda(lambda x: K.relu(x))(conv4_uCBCT)
    conv4_uCBCT = Conv3D(8 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4), name='uCBCT_conv4_2')(conv4_uCBCT)
    conv4_uCBCT = BatchNormalization(name='uCBCT_BN4_2')(conv4_uCBCT)
    conv4_uCBCT = Lambda(lambda x: K.relu(x))(conv4_uCBCT)
    conv4_uCBCT = Conv3D(8 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4), name='uCBCT_conv4_3')(conv4_uCBCT)
    conv4_uCBCT = BatchNormalization(name='uCBCT_BN4_3')(conv4_uCBCT)
    conv4_uCBCT = Lambda(lambda x: K.relu(x))(conv4_uCBCT)
    drop4_uCBCT = Dropout(dropout_rate)(conv4_uCBCT)
    pool4_uCBCT = MaxPooling3D(pool_size=(2, 2, 2))(drop4_uCBCT)

    conv5_uCBCT = Conv3D(16 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4), name='uCBCT_conv5_1')(pool4_uCBCT)
    conv5_uCBCT = BatchNormalization(name='uCBCT_BN5_1')(conv5_uCBCT)
    conv5_uCBCT = Lambda(lambda x: K.relu(x))(conv5_uCBCT)
    conv5_uCBCT = Conv3D(16 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4), name='uCBCT_conv5_2')(conv5_uCBCT)
    conv5_uCBCT = BatchNormalization(name='uCBCT_BN5_2')(conv5_uCBCT)
    conv5_uCBCT = Lambda(lambda x: K.relu(x))(conv5_uCBCT)
    conv5_uCBCT = Conv3D(16 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4), name='uCBCT_conv5_3')(conv5_uCBCT)
    conv5_uCBCT = BatchNormalization(name='uCBCT_BN5_3')(conv5_uCBCT)
    conv5_uCBCT = Lambda(lambda x: K.relu(x))(conv5_uCBCT)
    drop5_uCBCT = Dropout(dropout_rate)(conv5_uCBCT)

    up6 = Conv3D(8 * filter_base, 2, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                  bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                  kernel_regularizer=regularizers.l2(1e-4),
                  name='conv6_1')(UpSampling3D(size=(2, 2, 2))(drop5_uCBCT))
    up6 = BatchNormalization(name='BN6_1')(up6)
    up6 = Lambda(lambda x: K.relu(x))(up6)
    merge6 = Concatenate(axis=-1)([drop4_uCBCT, up6])
    conv6 = Conv3D(8 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    name='conv6_2')(merge6)
    conv6 = BatchNormalization(name='BN6_2')(conv6)
    conv6 = Lambda(lambda x: K.relu(x))(conv6)
    conv6 = Conv3D(8 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4),
                    name='conv6_3')(conv6)
    conv6 = BatchNormalization(name='BN6_3')(conv6)
    conv6 = Lambda(lambda x: K.relu(x))(conv6)

    up7 = Conv3D(4 * filter_base, 2, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                  bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                  kernel_regularizer=regularizers.l2(1e-4),
                  name='conv7_1')(UpSampling3D(size=(2, 2, 2))(conv6))
    up7 = BatchNormalization(name='BN7_1')(up7)
    up7 = Lambda(lambda x: K.relu(x))(up7)
    merge7 = Concatenate(axis=-1)([conv3_uCBCT, up7])
    conv7 = Conv3D(4 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4),
                    name='conv7_2')(merge7)
    conv7 = BatchNormalization(name='BN7_2')(conv7)
    conv7 = Lambda(lambda x: K.relu(x))(conv7)
    conv7 = Conv3D(4 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4),
                    name='conv7_3')(conv7)
    conv7 = BatchNormalization(name='BN7_3')(conv7)
    conv7 = Lambda(lambda x: K.relu(x))(conv7)

    up8 = Conv3D(2 * filter_base, 2, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                  bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                  kernel_regularizer=regularizers.l2(1e-4),
                  name='conv8_1')(UpSampling3D(size=(2, 2, 2))(conv7))
    up8 = BatchNormalization(name='BN8_1')(up8)
    up8 = Lambda(lambda x: K.relu(x))(up8)
    merge8 = Concatenate(axis=-1)([conv2_uCBCT, up8])
    conv8 = Conv3D(2 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4),
                    name='conv8_2')(merge8)
    conv8 = BatchNormalization(name='BN8_2')(conv8)
    conv8 = Lambda(lambda x: K.relu(x))(conv8)
    conv8 = Conv3D(2 * filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4),
                    name='conv8_3')(conv8)
    conv8 = BatchNormalization(name='BN8_3')(conv8)
    conv8 = Lambda(lambda x: K.relu(x))(conv8)

    up9 = Conv3D(filter_base, 2, padding='same',
                  kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                  bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                  kernel_regularizer=regularizers.l2(1e-4),
                  name='conv9_1')(UpSampling3D(size=(2, 2, 2))(conv8))
    up9 = BatchNormalization(name='BN9_1')(up9)
    up9 = Lambda(lambda x: K.relu(x))(up9)
    merge9 = Concatenate(axis=-1)([conv1_uCBCT, up9])
    conv9 = Conv3D(filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4),
                    name='conv9_2')(merge9)
    conv9 = BatchNormalization(name='BN9_2')(conv9)
    conv9 = Lambda(lambda x: K.relu(x))(conv9)
    conv9 = Conv3D(filter_base, 3, padding='same',
                    kernel_initializer=RandomNormal(mean=0.0, stddev=0.05, seed=None),
                    bias_initializer=RandomNormal(mean=1.0, stddev=0.05, seed=None),
                    kernel_regularizer=regularizers.l2(1e-4),
                    name='conv9_3')(conv9)
    conv9 = BatchNormalization(name='BN9_3')(conv9)
    conv9 = Lambda(lambda x: K.relu(x))(conv9)
    conv10 = Conv3D(1, 1, activation='relu', name='main_output')(conv9)
    
    # sino1 = kWave()(conv10) 

    model = Model(inputs=[input_uCBCT], outputs=[conv10])

    return model