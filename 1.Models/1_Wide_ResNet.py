import keras
import numpy as np
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers import Conv2D, Dense, Input, add, Activation, Flatten, AveragePooling2D
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.callbacks import ReduceLROnPlateau
from keras.regularizers import l2
from keras import optimizers
from keras.models import Model
from keras.models import load_model
from keras.utils import multi_gpu_model
import time
from focal_loss_LSR import focal_loss
from keras.callbacks import EarlyStopping

DEPTH              = 28
WIDE               = 10
IN_FILTERS         = 16

CLASS_NUM          = 10
IMG_ROWS, IMG_COLS = 32, 32
IMG_CHANNELS       = 3

BATCH_SIZE         = 128
EPOCHS             = 250
ITERATIONS         = 50000 // BATCH_SIZE + 1
WEIGHT_DECAY       = 0.0005
LOG_FILE_PATH      = './w_resnet/'


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

def scheduler(epoch):
    if epoch < 60:
        return 0.1
    if epoch < 120:
        return 0.02
    if epoch < 160:
        return 0.004
    if epoch < 200:
        return 0.0008
    if epoch < 230:
        return 0.0002
    return 0.00004

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


def wide_residual_network(img_input,classes_num,depth,k):
    print('Wide-Resnet %dx%d' %(depth, k))
    n_filters  = [16, 16*k, 32*k, 64*k]
    n_stack    = (depth - 4) // 6

    def conv3x3(x,filters):
        return Conv2D(filters=filters, kernel_size=(3,3), strides=(1,1), padding='same',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(WEIGHT_DECAY),
        use_bias=False)(x)

    def bn_relu(x):
        x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
        x = Activation('relu')(x)
        return x

    def residual_block(x,out_filters,increase=False):
        global IN_FILTERS
        stride = (1,1)
        if increase:
            stride = (2,2)
            
        o1 = bn_relu(x)
        
        conv_1 = Conv2D(out_filters,
            kernel_size=(3,3),strides=stride,padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(WEIGHT_DECAY),
            use_bias=False)(o1)

        o2 = bn_relu(conv_1)
        
        conv_2 = Conv2D(out_filters, 
            kernel_size=(3,3), strides=(1,1), padding='same',
            kernel_initializer='he_normal',
            kernel_regularizer=l2(WEIGHT_DECAY),
            use_bias=False)(o2)
        if increase or IN_FILTERS != out_filters:
            proj = Conv2D(out_filters,
                                kernel_size=(1,1),strides=stride,padding='same',
                                kernel_initializer='he_normal',
                                kernel_regularizer=l2(WEIGHT_DECAY),
                                use_bias=False)(o1)
            block = add([conv_2, proj])
        else:
            block = add([conv_2,x])
        return block

    def wide_residual_layer(x,out_filters,increase=False):
        global IN_FILTERS
        x = residual_block(x,out_filters,increase)
        IN_FILTERS = out_filters
        for _ in range(1,int(n_stack)):
            x = residual_block(x,out_filters)
        return x


    x = conv3x3(img_input,n_filters[0])
    x = wide_residual_layer(x,n_filters[1])
    x = wide_residual_layer(x,n_filters[2],increase=True)
    x = wide_residual_layer(x,n_filters[3],increase=True)
    x = BatchNormalization(momentum=0.9, epsilon=1e-5)(x)
    x = Activation('relu')(x)
    x = AveragePooling2D((8,8))(x)
    x = Flatten()(x)
    x = Dense(classes_num,
        activation='softmax',
        kernel_initializer='he_normal',
        kernel_regularizer=l2(WEIGHT_DECAY),
        use_bias=False)(x)
    return x

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
    y_train = keras.utils.to_categorical(y_train, CLASS_NUM)
    y_val = keras.utils.to_categorical(y_val, CLASS_NUM)
    y_test = keras.utils.to_categorical(y_test, CLASS_NUM)
    
    # color preprocessing
    x_train, x_test, x_val = color_preprocessing(x_train, x_test, x_val)

    # build network
    img_input = Input(shape=(IMG_ROWS,IMG_COLS,IMG_CHANNELS))
    output = wide_residual_network(img_input,CLASS_NUM,DEPTH,WIDE)
    resnet = Model(img_input, output)
    print(resnet.summary())

    
    # set optimizer
    # classes_num = []
    # s = all_np(y_train)
    # for i in range(len(s)):
    #     classes_num.append(s[i])
    # print(classes_num)     
    parallel_model = multi_gpu_model(resnet, gpus=2)
    parallel_model.compile(optimizer=optimizers.SGD(lr=.1, momentum=0.9, nesterov=True), loss='categorical_crossentropy', metrics=['accuracy'])
    # set callback
    tb_cb = TensorBoard(log_dir=LOG_FILE_PATH, histogram_freq=0)
    change_lr = LearningRateScheduler(scheduler)
    # lr_reducer = ReduceLROnPlateau(monitor='val_acc',factor=0.2,patience=5,
    #                            mode='max',min_lr=1e-3)
    cbks = [change_lr,tb_cb]

    # set data augmentation
    print('Using real-time data augmentation.')
    datagen = ImageDataGenerator(horizontal_flip=True,rotation_range=15,
            width_shift_range=0.125,height_shift_range=0.125,fill_mode='reflect')

    datagen.fit(x_train)

    # start training
    start = time.time()
    parallel_model.fit_generator(datagen.flow(x_train, y_train,batch_size=BATCH_SIZE),
                        steps_per_epoch=ITERATIONS,
                        epochs=EPOCHS,
                        callbacks=cbks,
                        validation_data=(x_val, y_val))

    
    loss,accuracy = parallel_model.evaluate(x_test,y_test)
    print('\ntest loss',loss)
    print('accuracy',accuracy)
    end = time.time()
    print('time',end-start)
    resnet.save('wresnet666.h5')
