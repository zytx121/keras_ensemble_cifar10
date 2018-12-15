import keras
import numpy as np
from scipy import stats
import pandas as pd
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, Flatten, AveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from keras import optimizers
from keras.models import Model, Input
from keras.models import load_model
from keras.utils import multi_gpu_model
import time
from focal_loss_LSR import focal_loss
from keras.callbacks import EarlyStopping

CLASS_NUM          = 10


from keras import backend as K
# set GPU memory 
if('tensorflow' == K.backend()):
    import tensorflow as tf
    from keras.backend.tensorflow_backend import set_session
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)


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


def color_preprocessing(x_train,x_test,x_val):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_val = x_val.astype('float32')
    mean = [125.3, 123.0, 113.9]
    std  = [63.0,  62.1,  66.7]
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
        x_val[:,:,:,i] = (x_val[:,:,:,i] - mean[i]) / std[i]

    return x_train, x_test, x_val


def all_np(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result




if __name__ == '__main__':

    # load data
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()
    print('Train data shape: ', x_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', x_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', x_test.shape)
    print('Test labels shape: ', y_test.shape)
    
    classes_num = []
    s = all_np(y_train)
    for i in range(len(s)):
        classes_num.append(s[i])
    print(classes_num)   
    y_val1 = np.squeeze(y_val)
    y_test1 = np.squeeze(y_test)
    y_train = keras.utils.to_categorical(y_train, CLASS_NUM)
    y_val = keras.utils.to_categorical(y_val, CLASS_NUM)
    y_test = keras.utils.to_categorical(y_test, CLASS_NUM)
    
    # color preprocessing
    x_train, x_test, x_val = color_preprocessing(x_train, x_test, x_val)

    # build network
    wresnet = load_model('wresnet666.h5')
    densenet = load_model('densenet.h5')
    resnext = load_model('resnext2.h5')
    senet = load_model('senet.h5')
    
    sgd = optimizers.SGD(lr=.1, momentum=0.9, nesterov=True)
    wresnet.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])   
    densenet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    resnext.compile(optimizer=sgd, loss=[focal_loss(classes_num)], metrics=['accuracy'])
    senet.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    print(senet.summary())
    # Predict labels with models
    dense_layer_model1 = Model(inputs=wresnet.input,
                                         outputs=wresnet.get_layer('dense_1').output)
    dense_layer_model2 = Model(inputs=densenet.input,
                                         outputs=densenet.get_layer('dense_1').output)
    dense_layer_model3 = Model(inputs=resnext.input,
                                         outputs=resnext.get_layer('dense_1').output)
    dense_layer_model4 = Model(inputs=senet.input,
                                         outputs=senet.get_layer('dense_19').output)

    dense_output1 = dense_layer_model1.predict(x_val)
    dense_output2 = dense_layer_model2.predict(x_val)
    dense_output3 = dense_layer_model3.predict(x_val)
    dense_output4 = dense_layer_model4.predict(x_val)


    best_error = 888
    best_renpin1 = 666
    best_renpin2 = 888
    best_renpin3 = 999

    for renpin1 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:  
        for renpin2 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: 
            for renpin3 in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]: 
                ams = (renpin1)*dense_output1+(renpin2)*dense_output2+(renpin3)*dense_output3+(3-renpin1-renpin2-renpin3)*dense_output4
                predicts = np.argmax(ams, axis=1)
                error = np.sum(np.not_equal(predicts, y_val1)) / y_val1.shape[0] 
                print(" Precision: {} , renpin: {} , renpin2: {} , renpin3: {}".format(1-error, renpin1, renpin2, renpin3))
                if error < best_error:
                    best_error = error
                    best_renpin1 = renpin1
                    best_renpin2 = renpin2
                    best_renpin3 = renpin3
    print("====================================================")            
    print("Best precision: {} , renpin1:  {} , renpin2: {} , renpin3: {}".format(1-best_error, best_renpin1, best_renpin2, best_renpin3))
    print("====================================================")
    test_output1 = dense_layer_model1.predict(x_test)
    test_output2 = dense_layer_model2.predict(x_test)
    test_output3 = dense_layer_model3.predict(x_test)
    test_output4 = dense_layer_model4.predict(x_test)
    ams1 = (best_renpin1)*test_output1+(best_renpin2)*test_output2+(best_renpin3)*test_output3+(3-best_renpin1-best_renpin2-best_renpin3)*test_output4
    predicts1 = np.argmax(ams1, axis=1)
    error1 = np.sum(np.not_equal(predicts1, y_test1)) / y_test1.shape[0] 
    print("Precision on test: {} , renpin1:  {} , renpin2: {} , renpin3: {}".format(1-error1, best_renpin1, best_renpin2, best_renpin3))