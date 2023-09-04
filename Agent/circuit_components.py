import cirq


def one_qubit_rotation(qubit, symbols):
    """ Function that applies rotation around x,y and z
        axis specified by the values in symbols.

    Args:
        qubit: The qubit to apply the rx, ry, rz gate to
        symbols: Array of syu
    """
    return [
        cirq.rx(symbols[0])(qubit),
        cirq.ry(symbols[1])(qubit),
        cirq.rz(symbols[2])(qubit)
    ]

def reg_entangling_layer(qubits):
    """
    Returns a layer of CZ entangling gates on `qubits` (arranged in a circular topology).
    """
    cz_ops = [cirq.CZ(q0, q1) for q0, q1 in zip(qubits, qubits[1:])]
    cz_ops += ([cirq.CZ(qubits[0], qubits[-1])] if len(qubits) != 2 else [])
    return cz_ops

def entangling_layer(qubits, part_of_hadamard_test=True):
    """ Returns a Cirq circuit that applies a layer of entangling gates (arranged in a circular topology).
        Args:
            qubits: list of qubits that are used in the circuit (last qubit is control qubit)
            part_of_hadamard_test: boolean that indicates whether the layer is part of the hadamard test
    """
    if len(qubits) == 2:
        yield cirq.Circuit(cirq.CZ(qubits[0], qubits[1]))
    else:
        yield cirq.Circuit(
            cirq.CZ(qubits[i], qubits[(i + 1) % len(qubits)])
            for i in range(len(qubits)))


def bravy_ghost_encoding(qubits, bravy_params):
    """ Function that adds a single bravi ghost encoding layer to the circuit
        Args:
            qubits: list of qubits that are used in the circuit (last qubit is control qubit)
            bravy_params: list of variational parameters for the bravyi gates
    """
    # Ghost encoding applied to the first qubit of the PQC
    yield cirq.ZPowGate(exponent=bravy_params[0],
                        global_shift=0)(qubits[0]).controlled_by(qubits[-1])

    # Ghost encoding applied to the last qubit of the PQC
    yield cirq.ZPowGate(exponent=bravy_params[1],
                        global_shift=0)(qubits[-2]).controlled_by(qubits[-1])
    # Apply swap gate to control qubit to switch circuit
    yield cirq.Circuit(cirq.X(qubits[-1]))

