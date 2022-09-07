import numpy as np
import os
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
from keras import metrics
import tensorflow as tf


import segmentation_models as sm
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def recall(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    all_positives = K.sum(K.round(K.clip(y_true, 0, 1)))

    recall = true_positives / (all_positives + K.epsilon())
    return recall


def precision(y_true, y_pred):
    y_true = K.ones_like(y_true)
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))

    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
        return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall)
    return f1_score


def focal_loss(gamma=2., alpha=.25):
    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -keras.sum(alpha * keras.pow(1. - pt_1, gamma) * keras.log(pt_1)) - keras.sum(
            (1 - alpha) * keras.pow(pt_0, gamma) * keras.log(1. - pt_0))

    return focal_loss_fixed


def focal_loss_2(gamma=2., alpha=.25):
    def focal_loss_2_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -keras.mean(alpha * keras.pow(1. - pt_1, gamma) * keras.log(pt_1)) - keras.mean(
            (1 - alpha) * keras.pow(pt_0, gamma) * keras.log(1. - pt_0))

    return focal_loss_2_fixed


def dice_loss(smooth=1.):
    def dice_loss_fixed(y_true, y_pred):
        intersection = K.sum(y_true * y_pred)
        union = K.sum(y_true) + K.sum(y_pred)
        return 1. - (2. * intersection + smooth) / (union + smooth)

    return dice_loss_fixed


def focal_dice_loss():
    gamma = 2.
    alpha = .25
    smooth = 1.

    def focal_loss_fixed(y_true, y_pred):
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
        return -keras.sum(alpha * keras.pow(1. - pt_1, gamma) * keras.log(pt_1)) - keras.sum(
            (1 - alpha) * keras.pow(pt_0, gamma) * keras.log(1. - pt_0))

    def dice_loss_fixed(y_true, y_pred):
        intersection = K.sum(y_true * y_pred)  # 计算X交Y，即TP
        union = K.sum(y_true) + K.sum(y_pred)  # 计算X和Y的和
        return 1. - (2. * intersection + smooth) / (union + smooth)

    def focal_dice_loss_fixed(y_pred, y_true, lambda_coef=0.5):
        return lambda_coef * focal_loss_fixed(y_true, y_pred) - K.log(dice_loss_fixed(y_true, y_pred))

    return focal_dice_loss_fixed


def tversky_loss(smooth=1.):
    def tversky_loss_fixed(y_true, y_pred):
        y_true_pos = K.flatten(y_true)
        y_pred_pos = K.flatten(y_pred)
        TP = K.sum(y_true_pos * y_pred_pos)
        FN = K.sum(y_true_pos * (1 - y_pred_pos))
        FP = K.sum((1 - y_true_pos) * y_pred_pos)
        alpha = 0.3  # alpha for FP and beta=(1-alpha) for FN
        tversky_coef = (TP + smooth) / (TP + alpha * FP + (1 - alpha) * FN + smooth)
        return 1. - tversky_coef

    return tversky_loss_fixed


# Dilated Convolutions & Focal Loss & BatchNormalization
def Crackdet(pretrained_weights=None, input_size=(512, 512, 3)):
    input = Input(input_size)

    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(input)
    conv1 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # conv1 = BatchNormalization()(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv111 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(pool1))

    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    # conv2 = BatchNormalization()(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv222 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(4, 4))(pool2))

    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # conv3 = BatchNormalization()(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv333 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(8, 8))(pool3))

    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # conv4 = BatchNormalization()(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv444 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(8, 8))(conv4))

    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    conv555 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(16, 16))(conv5))

    conv5 = Conv2D(512, 3, activation='relu', dilation_rate=(2, 2), padding='same', kernel_initializer='he_normal')( conv5)
    conv5 = Conv2D(512, 3, activation='relu', dilation_rate=(2, 2), padding='same', kernel_initializer='he_normal')( conv5)
    conv5 = Conv2D(512, 3, activation='relu', dilation_rate=(4, 4), padding='same', kernel_initializer='he_normal')( conv5)
    conv5 = Conv2D(512, 3, activation='relu', dilation_rate=(4, 4), padding='same', kernel_initializer='he_normal')(  conv5)
    conv5 = Conv2D(512, 3, activation='relu', dilation_rate=(8, 8), padding='same', kernel_initializer='he_normal')(  conv5)
    conv5 = Conv2D(512, 3, activation='relu', dilation_rate=(8, 8), padding='same', kernel_initializer='he_normal')( conv5)
    conv5 = Conv2D(512, 3, activation='relu', dilation_rate=(16, 16), padding='same', kernel_initializer='he_normal')( conv5)
    conv5 = Conv2D(512, 3, activation='relu', dilation_rate=(16, 16), padding='same', kernel_initializer='he_normal')( conv5)
    # conv5 = BatchNormalization()(conv5)

    pool5 = MaxPooling2D(pool_size=(32, 32))(conv5)
    conv6 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(pool5)
    up6 = UpSampling2D(size=(512, 512))(conv6)

    pool6 = MaxPooling2D(pool_size=(16, 16))(conv5)
    conv7 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(pool6)
    up7 = UpSampling2D(size=(256, 256))(conv7)

    pool7 = MaxPooling2D(pool_size=(8, 8))(conv5)
    conv8 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(pool7)
    up8 = UpSampling2D(size=(128, 128))(conv8)

    pool8 = MaxPooling2D(pool_size=(4, 4))(conv5)
    conv9 = Conv2D(128, 1, activation='relu', padding='same', kernel_initializer='he_normal')(pool8)
    up9 = UpSampling2D(size=(64, 64))(conv9)

    merge1 = concatenate([conv111, conv222, conv333, conv444, conv555, up6, up7, up8, up9], axis=3)
    # conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
    # conv9 = BatchNormalization()(conv9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge1)
    conv10 = Conv2D(32, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv11 = Conv2D(16, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv10)
    conv12 = Conv2D(8, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv11)
    conv13 = Conv2D(4, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv12)
    conv14 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv13)
    conv15 = Conv2D(1, 1, activation='sigmoid')(conv14)

    model = Model(inputs=input, outputs=conv15)

    print('model compile')
    # model.compile(optimizer = Adam(lr = 1e-5), loss = 'binary_crossentropy', metrics = ['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-5), loss=[tversky_loss(smooth=1.)], metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=5e-6), loss=[dice_loss(smooth=1.)], metrics=['accuracy'])
    model.compile(optimizer=Adam(lr=1e-4), loss=[dice_loss(smooth=1.)], metrics=['binary_accuracy',sm.metrics.IOUScore(threshold=0.5)])
  #  model.compile(optimizer=Adam(lr=1e-5), loss=[tversky_loss(smooth=1.)], metrics=['accuracy'])
    # model.compile(optimizer=Adam(lr=1e-5), loss=[focal_loss()], metrics=['accuracy'])

    model.summary()
    if (pretrained_weights):
        model.load_weights(pretrained_weights)
        print('loaded pretrained_weights ... {}'.format(pretrained_weights))

    return model

# U-net
def unet(pretrained_weights=None, input_size=(512, 512, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)  # 512
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv5))

    merge6 = concatenate([conv4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)


    model = Model(input=inputs, output=conv10)
    print('model compile')
    #model.compile(optimizer = Adam(lr = 1e-6), loss = 'binary_crossentropy',
    #model.compile(optimizer = Adam(lr = 1e-9), loss = [focal_loss(alpha=.25, gamma=2)],
                  # , metrics = ['accuracy'])
    # model.compile(optimizer = Adam(lr = 1e-4), loss = [focal_loss_2(alpha=.25, gamma=2)], metrics = ['accuracy'])
    model.compile(optimizer=Adam(lr=1e-6), loss=[dice_loss(smooth=1.)],
    #model.compile(optimizer=Adam(lr=1e-9), loss=[tversky_loss(smooth=1.)],
                  metrics=['binary_accuracy',recall,sm.metrics.IOUScore(threshold=0.5)])
    # model.compile(optimizer=Adam(lr=1e-4), loss=[tversky_loss(smooth=1.)], metrics=['accuracy',mean_iou,precision, recall,f1_score,'mae'])
    # model.compile(optimizer=Adam(lr=1e-5), loss=[focal_dice_loss()], metrics=['accuracy'])
    model.summary()
    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model

unet()




