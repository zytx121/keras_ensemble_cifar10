import keras
import math
import time
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, GlobalAveragePooling2D, multiply, Reshape
from keras.layers import Lambda, concatenate
from keras.initializers import he_normal
from keras.callbacks import LearningRateScheduler, TensorBoard, ModelCheckpoint
from keras.models import Model
from keras import optimizers
from keras.utils import multi_gpu_model
from keras import regularizers
from keras.applications.inception_v3 import InceptionV3



num_classes        = 10
batch_size         = 64  # 120       
iterations         = 782 # 416       # total data / iterations = batch size
epochs             = 100
epochs1             = 10

mean = [125.307, 122.95, 113.865]
std  = [62.9932, 62.0887, 66.7048]

from keras import backend as K
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
    
def scheduler(epoch):
    if epoch < 150:
        return 0.1
    if epoch < 225:
        return 0.01
    return 0.001

if __name__ == '__main__':

    # load data
    # (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    x_train, y_train, x_val, y_val, x_test, y_test = get_CIFAR10_data()
    
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    y_test  = keras.utils.to_categorical(y_test, num_classes)
    x_train = x_train.astype('float32')
    x_val = x_val.astype('float32')
    x_test  = x_test.astype('float32')
    
    # - mean / std
    for i in range(3):
        x_train[:,:,:,i] = (x_train[:,:,:,i] - mean[i]) / std[i]
        x_test[:,:,:,i] = (x_test[:,:,:,i] - mean[i]) / std[i]
        x_val[:,:,:,i] = (x_val[:,:,:,i] - mean[i]) / std[i]


    print('Train data shape before: ', x_train.shape)
    print('Validation data shape before: ', x_val.shape)
    print('Test data shape before: ', x_test.shape)
    print('Type train: ', type(x_train))
    
    x_train = tf.image.resize_images(x_train, [96, 96], method=0).eval(session = sess)
    x_test = tf.image.resize_images(x_test, [96, 96], method=0).eval(session = sess)
    x_val = tf.image.resize_images(x_val, [96, 96], method=0).eval(session = sess)
    
    print('Train data shape: ', x_train.shape)
    print('Validation data shape: ', x_val.shape)
    print('Test data shape: ', x_test.shape)
    print('Type train: ', type(x_train))
    # setting input pic
    input_img = Input(shape=(96, 96, 3)) 
    # create the base pre-trained model
    base_model = InceptionV3(input_tensor=input_img, weights='imagenet', include_top=False)

    # add a global spatial average pooling layer
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    # let's add a fully-connected layer
    x = Dense(1024, activation='relu')(x)
    # and a logistic layer -- let's say we have 200 classes
    predictions = Dense(10, activation='softmax')(x)

    # this is the model we will train
    model = Model(inputs=base_model.input, outputs=predictions)

    print(model.summary())

    # first: train only the top layers (which were randomly initialized)
    # i.e. freeze all convolutional InceptionV3 layers
    for layer in base_model.layers:
        layer.trainable = False

    # set optimizer
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
   

    # set callback
    tb_cb     = TensorBoard(log_dir='./Inception/', histogram_freq=0)                                   # tensorboard log
    # change_lr = LearningRateScheduler(scheduler)                                                    # learning rate scheduler
    # ckpt      = ModelCheckpoint('./ckpt_inception.h5', save_best_only=True, mode='auto', period=1)    # checkpoint 
    cbks      = [tb_cb]                   

    # set data augmentation
    print('Using real-time data augmentation.')

    datagen   = ImageDataGenerator(horizontal_flip=True,
            width_shift_range=0.125,height_shift_range=0.125,fill_mode='reflect')
    datagen.fit(x_train)

    # start training
    start = time.time()
    # parallel_model.fit(x_train, y_train,
    #       epochs=epochs,steps_per_epoch=iterations, callbacks=cbks,
    #       validation_data=(x_val, y_val),validation_steps=50)
    parallel_model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size), steps_per_epoch=iterations, epochs=epochs, callbacks=cbks,validation_data=(x_val, y_val))

    loss, accuracy = parallel_model.evaluate(x_test,y_test)
    print('\ntest loss',loss)
    print('accuracy',accuracy)
    end = time.time()
    print('transfer learning time',end-start)  
    model.save('transfer_inceptionV3.h5')


    # let's visualize layer names and layer indices to see how many layers
    # we should freeze:
    for i, layer in enumerate(base_model.layers):
       print(i, layer.name)

    # we chose to train the top 2 inception blocks, i.e. we will freeze
    # the first 249 layers and unfreeze the rest:
    for layer in model.layers[:249]:
       layer.trainable = False
    for layer in model.layers[249:]:
       layer.trainable = True

    # set optimizer
    # we need to recompile the model for these modifications to take effect
    # we use SGD with a low learning rate

    sgd = optimizers.SGD(lr=0.0001, momentum=0.9, nesterov=True)
    parallel_model = multi_gpu_model(model, gpus=2)
    parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
   

    # set callback
    tb_cb     = TensorBoard(log_dir='./Inception_finetune/', histogram_freq=0)                                   # tensorboard log
    # change_lr = LearningRateScheduler(scheduler)                                                    # learning rate scheduler
    # ckpt      = ModelCheckpoint('./ckpt_inception.h5', save_best_only=True, mode='auto', period=1)    # checkpoint 
    cbks      = [tb_cb]                   

    # set data augmentation
    print('Using real-time data augmentation.')

    datagen   = ImageDataGenerator(horizontal_flip=True,
            width_shift_range=0.125,height_shift_range=0.125,fill_mode='reflect')
    datagen.fit(x_train)

    # start training
    start = time.time()
    # parallel_model.fit(x_train, y_train,
    #       epochs=epochs,steps_per_epoch=iterations, callbacks=cbks,
    #       validation_data=(x_val, y_val),validation_steps=50)
    parallel_model.fit_generator(datagen.flow(x_train, y_train,batch_size=batch_size), steps_per_epoch=iterations, epochs=epochs1, callbacks=cbks,validation_data=(x_val, y_val))

    loss, accuracy = parallel_model.evaluate(x_test,y_test)
    print('\ntest loss',loss)
    print('accuracy',accuracy)
    end = time.time()
    print('fine tune time',end-start)  
    senet.save('finetune_inceptionV3.h5')



    
