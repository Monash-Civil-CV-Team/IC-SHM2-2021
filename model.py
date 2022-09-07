import numpy as np
import os
import segmentation_models.metrics
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
#import tensorflow.compat.v1 as tf
import tensorflow as tf
from keras import metrics
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D, AveragePooling2D
from keras.layers.core import Activation, Dropout, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.merge import add, concatenate
#from keras.optimizers import Adam
from keras import backend as K
import tensorflow as tf
def tversky_loss(smooth=1.):
    def tversky_loss_fixed(y_true, y_pred):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        TP = K.sum(y_true_pos * y_pred_pos)
        FN = K.sum(y_true_pos * (1-y_pred_pos))
        FP = K.sum((1-y_true_pos) * y_pred_pos)
        alpha = 0.3 # alpha for FP and beta=(1-alpha) for FN
        tversky_coef = (TP+smooth)/(TP+ alpha*FP +(1-alpha)*FN+smooth)
        return 1.-tversky_coef
    return tversky_loss_fixed
# dice loss defeind
def dice_loss(smooth=1.):
    def dice_loss_fixed(y_true, y_pred):
        intersection = K.sum(y_true * y_pred)
        union = K.sum(y_true)+K.sum(y_pred)
        return 1. -(2.*intersection+smooth)/(union+smooth)
    return dice_loss_fixed

def conv_block(input_tensor, filters, strides, d_rates):
    x = Conv2D(filters[0], kernel_size=1, dilation_rate=d_rates[0])(input_tensor)
  #  x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel_size=3, strides=strides, padding='same', dilation_rate=d_rates[1])(x)
  #  x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[2], kernel_size=1, dilation_rate=d_rates[2])(x)
 #   x = BatchNormalization()(x)

    shortcut = Conv2D(filters[2], kernel_size=1, strides=strides)(input_tensor)
 #   shortcut = BatchNormalization()(shortcut)

    x = add([x, shortcut])
    x = Activation('relu')(x)

    return x


def identity_block(input_tensor, filters, d_rates):
    x = Conv2D(filters[0], kernel_size=1, dilation_rate=d_rates[0])(input_tensor)
  #  x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[1], kernel_size=3, padding='same', dilation_rate=d_rates[1])(x)
 #   x = BatchNormalization()(x)
    x = Activation('relu')(x)

    x = Conv2D(filters[2], kernel_size=1, dilation_rate=d_rates[2])(x)
  #  x = BatchNormalization()(x)

    x = add([x, input_tensor])
    x = Activation('relu')(x)

    return x


def pyramid_pooling_block(input_tensor, bin_sizes):
    concat_list = [input_tensor]
    h = input_tensor.shape[1].value
    w = input_tensor.shape[2].value

    for bin_size in bin_sizes:
        x = AveragePooling2D(pool_size=(h//bin_size, w//bin_size), strides=(h//bin_size, w//bin_size))(input_tensor)
        x = Conv2D(512, kernel_size=1)(x)
        x = Lambda(lambda x: tf.image.resize_images(x, (h, w)))(x)

        concat_list.append(x)

    return concatenate(concat_list)


def aspp_block(input_tensor, num_filters, rate_scale=1):
    x1 = Conv2D(num_filters, (3, 3), dilation_rate=(1 * rate_scale, 1 * rate_scale), padding="same")(input_tensor)
  #  x1 = BatchNormalization()(x1)

    x2 = Conv2D(num_filters, (3, 3), dilation_rate=(2 * rate_scale, 2 * rate_scale), padding="same")(input_tensor)
 #   x2 = BatchNormalization()(x2)

    x3 = Conv2D(num_filters, (3, 3), dilation_rate=(3 * rate_scale, 3 * rate_scale), padding="same")(input_tensor)
 #   x3 = BatchNormalization()(x3)

    x4 = Conv2D(num_filters, (3, 3), padding="same")(input_tensor)
  #  x4 = BatchNormalization()(x4)

    x = Add()([x1, x2, x3, x4])
    x = Conv2D(num_filters, (1, 1), padding="same")(x)
    return x


def newmodel_noname(pretrained_weights=None, input_size=(512, 512, 3),multi_task=True,classes=[8,1,1,1,5]):
    input = Input(input_size)

    x = Conv2D(64, 3, strides=(2, 2),activation='relu', padding='same', kernel_initializer='he_normal')(input)
    x = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)

    x = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)

    x = MaxPooling2D((3, 3), strides=(2, 2), padding="same")(x)
    conv111 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(x))
    #print(conv111)
    x = conv_block(x, filters=[128, 128, 512], strides=(2, 2), d_rates=[1, 1, 1])
    x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])
    print(x)
    conv222 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(UpSampling2D(size=(2, 2))(x))

    x = conv_block(x, filters=[128, 128, 512], strides=(2, 2), d_rates=[1, 1, 1])
    x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])
    x=aspp_block(x,num_filters=1)
    x = conv_block(x, filters=[256, 256, 1024], strides=(1, 1), d_rates=[1, 1, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[256, 256, 1024], d_rates=[1, 1, 1])

    up2= Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(4, 4))(x))
    #print(up1)
    merge2 = concatenate([up2, conv222], axis=3)
    x = conv_block(merge2, filters=[128, 128, 512], strides=(1, 1), d_rates=[1, 1, 1])
    x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])
    x = identity_block(x, filters=[128, 128, 512], d_rates=[1, 1, 1])
    up1 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(x))
    #print(up1)
    merge1 = concatenate([up1, conv111], axis=3)
    x = pyramid_pooling_block(merge1, [1, 2, 3,6])




    #up3 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
    #    UpSampling2D(size=(4, 4))(x))


    #print(up3)
    #merge3 = concatenate([up3, x1], axis=3)

    x = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    print(x)

    #print(x)
    #x = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(x)
    #print(x)
    if multi_task == True:
       component = Conv2D(classes[0], (1, 1), padding='same')(x)
       component = Conv2DTranspose(classes[0], kernel_size=(16, 16), strides=(2, 2), padding='same')(component)
       component = Activation('softmax',name='component')(component)


       crack = Conv2D(classes[1], (1, 1), padding='same')(x)
       crack = Conv2DTranspose(classes[1], kernel_size=(16, 16), strides=(2, 2), padding='same')(crack )
       crack= Activation('sigmoid',name='crack')(crack )

       spall = Conv2D(classes[2], (1, 1), padding='same')(x)
       spall = Conv2DTranspose(classes[2], kernel_size=(16, 16), strides=(2, 2), padding='same')(spall)
       spall= Activation('sigmoid',name='spall')(spall )

       rebar = Conv2D(classes[3], (1, 1), padding='same')(x)
       rebar = Conv2DTranspose(classes[3], kernel_size=(16, 16), strides=(2, 2), padding='same')(rebar)
       rebar = Activation('sigmoid', name='rebar')(rebar)

       damage = Conv2D(classes[4], (1, 1), padding='same')(x)
       damage = Conv2DTranspose(classes[4], kernel_size=(16, 16), strides=(2, 2), padding='same')(damage)
       damage = Activation('softmax',name='damage')(damage)
       model = Model(input, outputs=[component,crack,spall,rebar,damage])
       model.compile(optimizer=Adam(lr=1e-4),
                     loss={'component': 'categorical_crossentropy', 'crack': dice_loss(smooth=1.),
                           'spall': dice_loss(smooth=1.),
                           'rebar': dice_loss(smooth=1.), 'damage': 'categorical_crossentropy'},
                     metrics=['categorical_accuracy', segmentation_models.metrics.IOUScore()])
    else:
       x = Conv2D(1, kernel_size=1)(x)
       x = Conv2DTranspose(1, kernel_size=(16, 16), strides=(2, 2), padding='same')(x)

       x = Activation('sigmoid')(x)

       model = Model(inputs=input, outputs=x)
       model.compile(optimizer=Adam(lr=1e-4),
                     loss=[dice_loss(smooth=1.)],
                     metrics=['binary_accuracy', segmentation_models.metrics.IOUScore()])
    #model = Model(img_input, x)
    #model.compile(optimizer=Adam(lr=lr_init, decay=lr_decay),
    #              loss='categorical_crossentropy',
    #              metrics=[dice_coef])
   # model.compile(optimizer=Adam(lr=1e-4), loss=[dice_loss(smooth=1.)], metrics=['accuracy'])




    model.summary()
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
        print('loaded pretrained_weights ... {}'.format(pretrained_weights))
    return model


newmodel_noname()






