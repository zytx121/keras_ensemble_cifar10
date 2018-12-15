import keras
import math
import time
import numpy as np
from keras.datasets import cifar10
# from focal_loss_LSR import focal_loss
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, AveragePooling2D, GlobalAveragePooling2D
from keras.layers import Lambda, concatenate
from keras.initializers import he_normal
from keras.layers.merge import Concatenate
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers
from keras import regularizers
from keras.utils import plot_model
from keras.utils import multi_gpu_model
import tensorflow as tf
from keras.models import *
from keras.layers import merge, Lambda
from keras import backend as K


# change the following para according to your GPUs.
gpu_number         = 2
batch_size         = 64         # 64 or 32 or smaller
epochs             = 250        
iterations         = 782       

growth_rate        = 24 
depth              = 164
compression        = 0.5

img_rows, img_cols = 32, 32
img_channels       = 3
num_classes        = 10
weight_decay       = 0.0001

mean = [125.307, 122.95, 113.865]
std  = [62.9932, 62.0887, 66.7048]

# set GPU memory 
if('tensorflow' == K.backend()):
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

def slice_batch(x, n_gpus, part):
    sh = K.shape(x)
    L =  sh[0] // n_gpus
    if part == n_gpus - 1:
        return x[part*L:]
    return x[part*L:(part+1)*L]



def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000):
    # Load the raw CIFAR-10 data
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    x_val = x_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    x_train = x_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    x_test = x_test[mask]
    y_test = y_test[mask]

    return x_train, y_train, x_val, y_val, x_test, y_test

def scheduler(epoch):
    if epoch <= 75:
        return 0.1
    if epoch <= 150:
        return 0.01
    if epoch <= 210:
        return 0.001
    return 0.0005


# def all_np(arr):
#     arr = np.array(arr)
#     key = np.unique(arr)
#     result = {}
#     for k in key:
#         mask = (arr == k)
#         arr_new = arr[mask]
#         v = arr_new.size
#         result[k] = v
#     return result


def densenet(img_input,classes_num):

    def bn_relu(x):
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x

    def bottleneck(x):
        channels = growth_rate * 4
        x = bn_relu(x)
        x = Conv2D(channels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
        x = bn_relu(x)
        x = Conv2D(growth_rate,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
        return x

    def single(x):
        x = bn_relu(x)
        x = Conv2D(growth_rate,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
        return x

    def transition(x, inchannels):
        outchannels = int(inchannels * compression)
        x = bn_relu(x)
        x = Conv2D(outchannels,kernel_size=(1,1),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(x)
        x = AveragePooling2D((2,2), strides=(2, 2))(x)
        return x, outchannels

    def dense_block(x,blocks,nchannels):
        concat = x
        for i in range(blocks):
            x = bottleneck(concat)
            concat = concatenate([x,concat], axis=-1)
            nchannels += growth_rate
        return concat, nchannels

    def dense_layer(x):
        return Dense(classes_num,activation='softmax',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay))(x)


    nblocks = (depth - 4) // 6 
    nchannels = growth_rate * 2

    x = Conv2D(nchannels,kernel_size=(3,3),strides=(1,1),padding='same',kernel_initializer=he_normal(),kernel_regularizer=regularizers.l2(weight_decay),use_bias=False)(img_input)

    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x, nchannels = dense_block(x,nblocks,nchannels)
    x, nchannels = transition(x,nchannels)
    x = bn_relu(x)
    x = GlobalAveragePooling2D()(x)
    x = dense_layer(x)
    return x


if __name__ == '__main__':

    # load data
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()
    print('Train labels shape: ', y_train.shape)
    
    # classes_num1 = []
    # s = all_np(y_train)
    # for i in range(len(s)):
    #     classes_num1.append(s[i])
    # print(classes_num1)

    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val  = keras.utils.to_categorical(y_val, num_classes)
    y_test  = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_val  = x_val.astype('float32')
    x_test  = x_test.astype('float32')
    
    # - mean / std
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_val[:,:,:,i] = (x_val[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]

    # build network
    img_input = Input(shape=(img_rows,img_cols,img_channels))
    output    = densenet(img_input,num_classes)
    model     = Model(img_input, output)
    print(model.summary())
    
    # -------------- Multi-GPU-------------------#
    parallel_model = multi_gpu_model(model, gpus=2)
    # -------------- Multi-GPU-------------------#

    # parallel_model.load_weights('ckpt.225-0.9580.h5')

    
    

    # set optimizer
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])

    
    # set callback
    tb_cb     = TensorBoard(log_dir='./densenet/', histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    ckpt      = ModelCheckpoint('./ckpt.{epoch:02d}-{val_acc:.4f}.h5', save_best_only=True, mode='auto', period=25)
    cbks      = [change_lr,tb_cb,ckpt]

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen   = ImageDataGenerator(horizontal_flip=True,width_shift_range=0.125,height_shift_range=0.125,fill_mode='constant',cval=0.)

    datagen.fit(x_train)

    # start training
    start = time.time()
    parallel_model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size), steps_per_epoch=iterations, epochs=epochs, callbacks=cbks,validation_data=(x_val, y_val))
    loss,accuracy = parallel_model.evaluate(x_test,y_test)
    print('\ntest loss',loss)
    print('accuracy',accuracy)
    end = time.time()
    print('time',end-start)     
    model.save('densenet.h5')