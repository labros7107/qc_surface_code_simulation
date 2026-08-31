def _create_matrix(bit_string, d=3):

    mtx = []

    row = []

    for i in bit_string.split(" ")[0]:

        if len(row) == d:
            mtx.append(row)
            row = []


        row.append(int(i))

    mtx.append(row)
    return mtx



def measure_parity(bit_string, d=3, basis="z"):

    matrix = _create_matrix(bit_string, d)

    if basis == "z":

        row_parity = set([(sum(row) % 2) for row in matrix])

        if len(row_parity) == 1:
            return row_parity

        else:
            raise ValueError("Parity is not consistent")


    if basis == "x":


        col_parity = set([(sum(col) % 2) for col in zip(*matrix)])

        if len(col_parity) == 1:
            return col_parity

        else:
            raise ValueError("Parity is not consistent")
