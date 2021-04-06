import tensorflow as tf
import tensorflow.keras as k

from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.applications import VGG16 as vgg16
from tensorflow.keras.applications.vgg16 import preprocess_input


def pre_vgg(layers):
        
    vgg = vgg16(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layers]
    model = k.Model([vgg.input], outputs)

    return model


class FCN(Model):

    def __init__(self):
        super(FCN, self).__init__(name="FCN_Model")
    
    def call(self, inputs):