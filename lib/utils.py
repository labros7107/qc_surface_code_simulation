import numpy as np


def get_data_qubits(n, qubit_list: np.ndarray) -> np.ndarray:

    qubit_count = -1*((n*n)+(n-1)*(n-1))

    return qubit_list[qubit_count:]


def print_parity(string):

    # print(data_qubits)
    total = len(string)
    ones = np.count_nonzero(string)
    zeros = total - ones
    # print(string)
    parity = np.sum(string) % 2
    print(f"ones:{ones} zeros: {zeros} parity:{parity}")
