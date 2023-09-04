"""Module providing the circuit generator for the pqx model"""
from functools import reduce

import cirq
import numpy as np
import tensorflow as tf
import tensorflow_quantum as tfq
from tensorflow.keras.constraints import Constraint

#from red_partition_circuit_gen import CircuitGenerator
from Agent.red_partition_circuit_gen import CircuitGenerator


class NonNegConstraint(Constraint):
    def __call__(self, w):
        return tf.nn.relu(w)

class CumToOne(Constraint):
    def __call___(self, w):
        return 

class ReducedPartitionPQCLayer(tf.keras.layers.Layer):
    """ This class is responsible for generating the PQC model.
        Args:
            n: number of qubits in the PQC
            d: depth of the PQC
            K: number of partitions
            L: number of terms in the sum
    """
    def __init__(self,
                 n_qubits, 
                 depth,
                 n_partitions,
                 n_terms,
                 input_dim,
                 trainable_lambdas,
                 rescaling_scheme,
                 trainable_regular_weights=True,
                 trainable_partition_weights=True,
                 name="re-uploading_PQC"
            ) -> None:
        super().__init__(name=name)
        #print("trainable_regular_weights:", trainable_regular_weights)
        self.n_qubits = n_qubits
        self.depth = depth
        self.n_partitions = n_partitions
        self.rescaling_scheme = rescaling_scheme
        self.n_terms = n_terms # T in the paper, product of schmidt number squared with gate cuts
                               # In our case for the CZ Gate 4 * gate cuts
        self.input_dim = input_dim
        # self.rescale_parameter = rescale_parameter
        # define qubits, observables
        qubits = cirq.GridQubit.rect(1, self.n_qubits+1)
        ops =  [cirq.Z(q) for q in qubits]
        observables = [reduce((lambda x, y: x*y), ops)]

        # define circuits
        generator = CircuitGenerator(qubits, self.depth)
        circuit, rotation_params, bravy_params, input_params = generator.generate_circuit(qubits)
        circuit_i, _, _, _, = generator.generate_circuit(qubits, real=False)

        self.reference_circuit = circuit
    
        # Initialize weights
        theta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        self.theta = tf.Variable(
            initial_value=theta_init(shape=(1, len(rotation_params)*n_partitions), dtype="float32"),
            trainable=trainable_regular_weights, name="thetas"
        )
        self.product_term_theta_size = len(rotation_params) # store length of rotation params
        
        # Weights to multiply by the training data (input encodings)
        alphas_init = tf.ones(shape=(len(input_params)*n_partitions,))
        self.alphas = tf.Variable(
            initial_value=alphas_init, dtype="float32",
            trainable=trainable_regular_weights, name="alphas"
        )
        self.input_dim = len(input_params) * n_partitions

        # Weights for bravy parameters
        zeta_init = tf.random_uniform_initializer(minval=0.0, maxval=np.pi)
        #print("length bravy", len(bravy_params))
        self.zeta = tf.Variable(
            initial_value=zeta_init(shape=(n_terms, n_partitions, len(bravy_params)),
            dtype="float32"), trainable=trainable_partition_weights, name="zetas"
        )


        self.activation = "linear"
        self.empty_circuit = tfq.convert_to_tensor([cirq.Circuit()])
        self.computation_layer = tfq.layers.ControlledPQC(circuit, observables)
        self.computation_layer_i = tfq.layers.ControlledPQC(circuit_i, observables)
        self.flatten = tf.keras.layers.Flatten(input_shape=(input_dim, input_dim))

        symbols = [str(symb) for symb in rotation_params + input_params + bravy_params]
        self.indices = tf.constant([symbols.index(a) for a in sorted(symbols)])

        rescale_parameter = 1 # We normalize the input vector so this should probably be initialized somewhere around here.
        lambdas_init = tf.ones(shape=(n_terms,)) * rescale_parameter

        if self.rescaling_scheme in ["constant", "factoring"]: # Here we take the lambda rescaling scheme without the exponentials
            self.lambdas = tf.Variable(initial_value=lambdas_init, dtype="float32",
                                        trainable=trainable_partition_weights, name="lambdas", constraint=NonNegConstraint())
        else : # Exponential facotoring
            self.lambdas = tf.Variable(initial_value=lambdas_init, dtype="float32",
                                        trainable=trainable_partition_weights, name="lambdas")
      
    
    # Funtion that returns zetas
    def get_zetas(self):
        return self.get_zetas

    # Function that returns indices
    def get_indices(self):
        return self.indices


    # Model call function
    def rescale_lambdas(self, inputs):
        """ This fucntion is responsible for the rescaling of the lambda parameters """
        batch_dim = tf.gather(tf.shape(inputs[0]), 0) # Gather the amount of datapoints
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1]) # Replicate inputs batch_dim times

        inputs = [self.flatten(inputs[0])]
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.depth])
        scaled_inputs = tf.einsum("i,ji->ji", self.alphas, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)
        
        # Simple rescaling of the lambdas
        for i in range(self.n_terms):
            self.lambdas[i].assign(1/self.n_terms)
        
        tf.print("rescaled lambdas:", self.lambdas)

    
    # Model call function
    def call(self, inputs):
        """ This fucntion is responsible for """
        batch_dim = tf.gather(tf.shape(inputs[0]), 0) # Gather the amount of datapoints
        tiled_up_thetas = tf.tile(self.theta, multiples=[batch_dim, 1]) # Replicate inputs batch_dim times
        inputs = [self.flatten(inputs[0])]
        tiled_up_inputs = tf.tile(inputs[0], multiples=[1, self.depth])
        scaled_inputs = tf.einsum("i,ji->ji", self.alphas, tiled_up_inputs)
        squashed_inputs = tf.keras.layers.Activation(self.activation)(scaled_inputs)

        #all_lambdas = self.lambdas

        ans = tf.zeros([batch_dim, 1]) # Initialize answer
        for i in range(self.n_terms): # Number of terms in product.
            # Define complex number for product part.
            pqc_layer_ans = tf.complex(tf.ones([batch_dim, 1]), tf.zeros([batch_dim, 1]))
            for k in range(self.n_partitions): # Number of terms in product (number of partitions)
                # get circuits
                tiled_up_circuits = tf.repeat(self.empty_circuit, repeats=batch_dim)
                tiled_up_circuits_i = tf.repeat(self.empty_circuit, repeats=batch_dim)
                # Get the current inputs for this term
                current_inputs = tf.slice(squashed_inputs, [0, k*self.n_qubits],
                                          [batch_dim, self.n_qubits*self.depth])
                # Get our thetas
                current_thetas = tf.slice(tiled_up_thetas, [0, k*self.product_term_theta_size],
                                         [batch_dim, self.product_term_theta_size])
                # Get our zetas
                current_zetas = tf.slice(self.zeta, [i, k, 0], [1, 1, -1])
                # Tile our zetas
                tiled_zetas = tf.tile(tf.reshape(current_zetas, current_zetas.shape[1:]),
                                                 multiples=[batch_dim, 1])

                # Batch our parameters for model training
                batched_params = tf.concat(values=[current_thetas, current_inputs, tiled_zetas], axis=1)
                batched_params = tf.gather(batched_params, self.indices, axis=1)

                computation = tf.complex(self.computation_layer([tiled_up_circuits, batched_params]),
                                         self.computation_layer_i([tiled_up_circuits_i, batched_params]))
                # Calculate the value of the current product term
                pqc_layer_ans = tf.multiply(computation, pqc_layer_ans)
       

            non_exp_lambda = self.lambdas[i]
            exp_lambdas = tf.math.exp(non_exp_lambda) # get the lambdas for this term in the product
            pqc_layer_ans = tf.math.real(pqc_layer_ans) # get the real part of the product term
           
            if self.rescaling_scheme in ["constant", "factoring"]:
                ans = tf.math.add(non_exp_lambda * pqc_layer_ans, ans)
            else:
                ans = tf.math.add(exp_lambdas * pqc_layer_ans, ans) # add the product term to the total ans
        
        return ans


