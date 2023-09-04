import cirq
import numpy as np
import tensorflow as tf
from keras import backend as K

from Agent.alternating_layer import Alternating
from Agent.pqc_layer_gen import PQCLayer
# Import model layers
from Agent.red_partition_layer_gen import ReducedPartitionPQCLayer


class Part_ReuploadingPQC(tf.keras.Model):
    """ This class is responsible for generating the PQC model"""
    def __init__(self,
                 n_qubits,
                 n_layers,
                 n_actions,
                 n_partitions,
                 n_terms,
                 input_dim,
                 beta,
                 trainable_lambdas,
                 rescaling_scheme,
                 trainable_regular_weights=True,
                 trainable_partition_weights=True,
                ):
        super(Part_ReuploadingPQC, self).__init__()
        self.n_qubits = n_qubits
        self.n_partitions = n_partitions
        self.partitioned_pqc = ReducedPartitionPQCLayer(
            n_qubits, n_layers, n_partitions, n_terms, input_dim,
            trainable_lambdas, rescaling_scheme, trainable_regular_weights, 
            trainable_partition_weights
        ) 
    
        self.alternating = Alternating(n_actions) # action specific expectation values
        self.lambda_layer = tf.keras.layers.Lambda(lambda x: x * beta) # Inverse temperature
        self.softmax = tf.keras.layers.Softmax() # Softmax

    def rescale_lambdas(self, state_bounds):
        self.partitioned_pqc.rescale_lambdas(state_bounds)

    def call(self, inputs):
        x = self.partitioned_pqc([inputs])
        x = self.alternating(x)
        x = self.lambda_layer(x)
        x = self.softmax(x)
        return x

    def call_presoftmax(self, inputs):
        """This is to analyze the pre-softmax model outputs"""
        x = self.partitioned_pqc([inputs])
        x = self.alternating(x)
        x = self.lambda_layer(x)
        return x
    
    def call_pqc(self, inputs):
        x = self.partitioned_pqc([inputs])
        return x
    
    def model(self):
        x = tf.keras.Input(shape=(self.n_qubits*self.n_partitions,))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

class ReuploadingPQC(tf.keras.Model):
    """ This class is responsible for generating the PQC model"""
    def __init__(self, n_qubits, n_layers, n_actions, beta):
        super(ReuploadingPQC, self).__init__()
        self.re_uploading_pqc = PQCLayer(n_qubits, n_layers)
        self.alternating = Alternating(n_actions)
        self.lambda_layer = tf.keras.layers.Lambda(lambda x: x * beta)
        self.softmax = tf.keras.layers.Softmax()
    
    def call(self, inputs):
        x = self.re_uploading_pqc([inputs])
        x = self.alternating(x)
        x = self.lambda_layer(x)
        x = self.softmax(x)
        return x
        
    def call_presoftmax(self, inputs):
        x = self.re_uploading_pqc([inputs])
        x = self.alternating(x)
        x = self.lambda_layer(x)
        return x
    
    def model(self):
        x = tf.keras.Input(shape=(self.n_qubits,))
        return tf.keras.Model(inputs=[x], outputs=self.call(x))

    
