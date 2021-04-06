import tensorflow as tf
import tensorflow.keras as k

from net import *

# Set GPU Operation Growth
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)