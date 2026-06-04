import tensorflow as tf
from tensorflow.keras import layers

@tf.keras.utils.register_keras_serializable(package="Custom")
class CustomAttention(layers.Layer):

    def __init__(self, **kwargs):
        super(CustomAttention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.W = self.add_weight(
            name="att_weight",
            shape=(input_shape[-1], 1),
            initializer="glorot_uniform",
            trainable=True
        )
        self.b = self.add_weight(
            name="att_bias",
            shape=(input_shape[1], 1),
            initializer="zeros",
            trainable=True
        )
        super(CustomAttention, self).build(input_shape)

    def call(self, inputs):
        e = tf.tensordot(inputs, self.W, axes=[-1, 0]) + self.b 
        e = tf.nn.tanh(e)
        alpha = tf.nn.softmax(e, axis=1)
        context = inputs * alpha 
        context = tf.reduce_sum(context, axis=1) 
        return context

    def get_config(self):
        config = super(CustomAttention, self).get_config()
        return config
