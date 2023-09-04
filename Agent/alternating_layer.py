import tensorflow as tf


class Alternating(tf.keras.layers.Layer):
    """ This class is responsible for the alternating layer of the model """
    def __init__(self, output_dim):
        super(Alternating, self).__init__()
        self.w = tf.Variable(
            initial_value=tf.constant([[(-1.)**i for i in range(output_dim)]]),
            dtype="float32",
            trainable=True,
            name="obs-weights"
        )

    def call(self, inputs):
        return tf.matmul(inputs, self.w)
