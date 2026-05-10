import numpy as np


def get_data_qubits(n, qubit_list: np.ndarray) -> np.ndarray:

    qubit_count = -1*((n*n)+(n-1)*(n-1))

    return qubit_list[qubit_count:]


def print_parity(data_qubits, s=int, f=int):

    # print(data_qubits)
    total = len(data_qubits)
    ones = np.count_nonzero(data_qubits)
    zeros = total - ones
    string = data_qubits[s:f]
    print(string)
    parity = np.sum(string) % 2
    print(f"ones:{ones} zeros: {zeros} parity:{parity}")
