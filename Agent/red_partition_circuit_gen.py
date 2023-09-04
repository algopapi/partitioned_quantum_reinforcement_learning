""" Generator of individual circuits that are required for the partition scheme """

import cirq
import numpy as np
import sympy
#Visualization
from cirq.contrib.svg import SVGCircuit

import Agent.circuit_components as cc


class CircuitGenerator():
    """ Class that generates a partitioned PQC circuit.

    Attributes:
        qubits: list of qubits that are used in the circuit
        n_layers: depth of the PQC
    """

    def __init__(self, qubits, n_layers) -> None:
        self.qubits = qubits  # list of qubits where the last qubit is the control qubit (split size)
        self.n_qubits = len(qubits) - 1  # number of qubits in each partition (without contorl qubit)
        self.n_layers = n_layers  # number of layers of the partitioned pqc (depth)

    def generate_layer(self, layer, qubits, rotation_params, bravy_params,
                       input_params):
        """ Generates a single layer of the partitioned PQC.
        Args:
            qubits: list of qubits that are used in the circuit (qubit[n] = control qubit)
            layer: the layer number of the PQC
            rotation_params: list of variational parameters for the rotation gates
            input_params: list of input parameters for the input encoding
            bravy_params: list of variational parameters for the bravyi gates

        Returns:
            layer: the generated circuit of a single pqc layer

        """
        circuit = cirq.Circuit()
        # Add a block of rotational gates to the layer
        circuit += cirq.Circuit(
            cc.one_qubit_rotation(q, rotation_params[i])
            for i, q in enumerate(qubits[:-1]))
        # Add the etangling layer to the circuit
        circuit += cc.entangling_layer(qubits[:-1])
        # Add first bravi ghost encoding
        circuit += cc.bravy_ghost_encoding(qubits, bravy_params[layer])
        # Add second bravi ghost encoding
        circuit += cc.bravy_ghost_encoding(qubits,
                                           bravy_params[layer + self.n_layers])
        # Input Encoding Layer
        circuit += cirq.Circuit(
            cirq.rx(input_params[i])(qubit)
            for i, qubit in enumerate(qubits[:-1]))
        #print("Layer_{} \n", layer, circuit)
        return circuit

    def generate_circuit(self, qubits, real=True):
        """ Function that combines the previously generated partitioned layers into the
            required circuits needed for evaluation of the PQC.
        """
        # Generate parameters for variational angles for the entire circuit
        theta_params = sympy.symbols(
            f'theta(0:{3*((self.n_layers+1) * self.n_qubits)})')
        theta_params = np.asarray(theta_params).reshape(
            (self.n_layers + 1, self.n_qubits, 3))
       
        # Generate parameters for the bravyi gates
        bravi_params = np.asarray(
            sympy.symbols(f'zeta(0:{4*self.n_layers})')).reshape(
                (2 * self.n_layers, 2))
        
        # Generate parameters for the input encoding
        inputs = sympy.symbols(f'x(0:{self.n_layers})' + f'_(0:{self.n_qubits})')
        inputs = np.asarray(inputs).reshape((self.n_layers, self.n_qubits))

        # Generate full circuit
        circuit = cirq.Circuit()

        # Apply hadamard gate to control qubit
        circuit += cirq.Circuit(cirq.H(qubits[-1]))
        if not real:
            circuit += cirq.S(qubits[-1])
        
        # Add layers to the circuit model
        for layer in range(self.n_layers):
            circuit += self.generate_layer(layer, qubits, theta_params[layer],
                                           bravi_params, inputs[layer])
        # Add final rotational layer block
        circuit += cirq.Circuit(
            cc.one_qubit_rotation(qubit, theta_params[-1, i])
            for i, qubit in enumerate(qubits[:-1]))
        
        # Add final hadamard gate to reinvert the circuit
        circuit += cirq.Circuit(cirq.H(qubits[-1]))
        if not real:
            circuit += cirq.S(qubits[-1])

        #print("full circuit \n", circuit)
        return circuit, list(theta_params.flat), list(bravi_params.flat), list(inputs.flat)
    