import os
import tensorflow as tf
from tensorflow.kears import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from fr_utils import triplet_loss
from config import input_shape
from model import build_model
from data_sequence_generator import *

# Calbacks
model_checkpoint_path = './model/best_model.h5'
checkPoint_Callback = tf.keras.callbacks.ModelCheckpoint(
    filepath = model_checkpoint_path,
    save_weights_only=True,
    monitor='val_loss',
    mode='min',
    verbose = 1,
    save_best_only=True
)

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=20,
                    verbose=1,
                    mode='auto',
                    min_delta=0.0001,
                    cooldown=0,
                    min_lr=0,
                )


adagrad = tf.keras.optimizers.Adagrad(
            learning_rate=0.001,
            initial_accumulator_value=0.1,
            epsilon=1e-07,
            name='Adagrad',
        )

# Base model
base_model = build_model()

# Model
def FRmodel():
    A = Input(shape=input_shape, name = 'anchors')
    P = Input(shape=input_shape, name = 'positives')
    N = Input(shape=input_shape, name = 'negatives')

    enc_A = base_model(A)
    enc_P = base_model(P)
    enc_N = base_model(N)

    output = concatenate([enc_A, enc_P, enc_N])

    FRmodel = Model(inputs=[A, P, N], outputs=output)
    
    return FRmodel


model = FRmodel()

model.compile(loss = triplet_loss, optimizer = adagrad)

if os.path.exists('./model/best_model.h5'):
    model.load_weights('./model/best_model.h5')

history = model.fit(
    DataSequence('train'),
    validation_data=DataSequence('val'),
    verbose=1,
    epochs=300,
    callbacks = [checkPoint_Callback, reduce_lr]
)