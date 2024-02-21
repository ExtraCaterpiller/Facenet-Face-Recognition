import tensorflow as tf
import numpy as np
import pickle
from tensorflow.keras.layers import Conv2D, Activation, ZeroPadding2D, BatchNormalization
from config import embedding_size


def triplet_loss(y_true, y_pred, alpha=0.1):
    anchor, positive, negative = y_pred[:, 0:embedding_size], y_pred[:, embedding_size:2*embedding_size], y_pred[:, 2*embedding_size:3*embedding_size]
    """Calculate the triplet loss according to the FaceNet paper
    
    Args:
      anchor: the embeddings for the anchor images.
      positive: the embeddings for the positive images.
      negative: the embeddings for the negative images.
  
    Returns:
      the triplet loss according to the FaceNet paper as a float tensor.
    """
    pos_dist = tf.math.reduce_sum(tf.math.square(tf.math.subtract(anchor, positive)), -1)
    neg_dist = tf.math.reduce_sum(tf.math.square(tf.math.subtract(anchor, negative)), -1)
        
    basic_loss = tf.math.add(tf.math.subtract(pos_dist,neg_dist), alpha)
    loss = tf.math.maximum(basic_loss, 0.0)
      
    return loss


def conv2d_bn(x,
              layer=None,
              cv1_out=None,
              cv1_filter=(1, 1),
              cv1_strides=(1, 1),
              cv2_out=None,
              cv2_filter=(3, 3),
              cv2_strides=(1, 1),
              padding=None):
    num = '' if cv2_out == None else '1'
    tensor = Conv2D(cv1_out, cv1_filter, strides=cv1_strides, data_format='channels_last', name=layer+'_conv'+num)(x)
    tensor = BatchNormalization(axis=-1, epsilon=0.00001, name=layer+'_bn'+num)(tensor)
    tensor = Activation('relu')(tensor)
    if padding == None:
        return tensor
    tensor = ZeroPadding2D(padding=padding, data_format='channels_last')(tensor)
    if cv2_out == None:
        return tensor
    tensor = Conv2D(cv2_out, cv2_filter, strides=cv2_strides, data_format='channels_last', name=layer+'_conv'+'2')(tensor)
    tensor = BatchNormalization(axis=-1, epsilon=0.00001, name=layer+'_bn'+'2')(tensor)
    tensor = Activation('relu')(tensor)
    return tensor



def load_file(file_path, type_of_image):
    with open(file_path, 'rb') as file:
        loaded_data = pickle.load(file)
    if type_of_image == 'anchor':
        return loaded_data['anchor']
    if type_of_image == 'positive':
        return loaded_data['positive']
    else:
        return loaded_data['negative']
    