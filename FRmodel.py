from tensorflow.keras import Input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import concatenate
from model import build_model
from config import image_shape

def FRmodel():

    base_model = build_model()

    A = Input(shape=image_shape, name = 'anchors')
    P = Input(shape=image_shape, name = 'positives')
    N = Input(shape=image_shape, name = 'negatives')

    enc_A = base_model(A)
    enc_P = base_model(P)
    enc_N = base_model(N)

    output = concatenate([enc_A, enc_P, enc_N])

    FRmodel = Model(inputs=[A, P, N], outputs=output)
    
    return FRmodel

def get_faceRecoModel():
    frmodel = FRmodel()
    
    frmodel.load_weights('./model/best_model.h5')

    inner_model_name = 'FaceRecoModel'
    inner_model = frmodel.get_layer(inner_model_name)

    model = Model(inputs=inner_model.input, outputs=inner_model.output)
    
    return model