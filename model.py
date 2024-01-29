from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, AveragePooling2D, Activation, ZeroPadding2D, BatchNormalization, Lambda
from tensorflow.keras import Input
from inception_blocks_v2 import *
from config import image_shape


"""
    Implementation of the Inception model used for FaceNet
    
    Arguments:
    input_shape -- shape of the images of the dataset

    Returns:
    model -- a Model() instance in Tensorflow-Keras
"""


def build_model():
    # Define the input as a tensor with shape input_shape
    X_input = Input(image_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3), data_format='channels_last')(X_input)
    
    # First Block
    X = Conv2D(64, (7, 7), strides = (2, 2), data_format='channels_last', name = 'conv1')(X)
    X = BatchNormalization(axis = -1, name = 'bn1')(X)
    X = Activation('relu')(X)
    
    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1), data_format='channels_last')(X)
    X = MaxPooling2D((3, 3), strides = 2, data_format='channels_last')(X)
    
    # Second Block
    X = Conv2D(64, (1, 1), strides = (1, 1), data_format='channels_last', name = 'conv2')(X)
    X = BatchNormalization(axis = -1, epsilon=0.00001, name = 'bn2')(X)
    X = Activation('relu')(X)
    
    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1), data_format='channels_last')(X)

    # Second Block
    X = Conv2D(192, (3, 3), strides = (1, 1), data_format='channels_last', name = 'conv3')(X)
    X = BatchNormalization(axis = -1, epsilon=0.00001, name = 'bn3')(X)
    X = Activation('relu')(X)
    
    # Zero-Padding + MAXPOOL
    X = ZeroPadding2D((1, 1), data_format='channels_last')(X)
    X = MaxPooling2D(pool_size = 3, strides = 2, data_format='channels_last')(X)
    
    # Inception 1: a/b/c
    X = inception_block_1a(X)
    X = inception_block_1b(X)
    X = inception_block_1c(X)
    
    # Inception 2: a/b
    X = inception_block_2a(X)
    X = inception_block_2b(X)
    
    # Inception 3: a/b
    X = inception_block_3a(X)
    X = inception_block_3b(X)
    
    # Top layer
    X = AveragePooling2D(pool_size=(3, 3), strides=(1, 1), data_format='channels_last')(X)
    X = Flatten(data_format='channels_last')(X)
    X = Dense(128, name='dense_layer')(X)
    
    # L2 normalization
    X_output = Lambda(lambda  x: tf.math.l2_normalize(x,axis=-1))(X)

    # Create model instance
    model = Model(inputs = X_input, outputs = X_output, name='FaceRecoModel')
    
    return model