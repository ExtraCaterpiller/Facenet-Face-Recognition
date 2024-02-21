import tensorflow as tf
from fr_utils import triplet_loss
from data_sequence_generator import *
from FRmodel import FRmodel

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

# Model
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