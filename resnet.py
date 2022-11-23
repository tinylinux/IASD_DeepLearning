import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class ResidualGo(layers.Layer):
    """docstring for ResidualGo."""

    def __init__(self, arg):
        super(Residual, self).__init__()
        self.arg = arg
