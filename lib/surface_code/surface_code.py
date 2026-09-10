# -*- coding: utf-8 -*-

# This code is part of Qiskit.
#
# (C) Copyright IBM 2019.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

# pylint: disable=invalid-name

"""Generates circuits based on repetition codes."""

import sys
import os

# Add the target directory to the search path
sys.path.append(os.path.abspath('/home/samir/research/experiments/surface_code_simulations'))

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister

from lib.surface_code.decoding_graph import DecodingGraphNode


class SurfaceCodeCircuit():
    """Distance d rotated surface code with  T syndrome measurement rounds."""

    def __init__(self, d: int, T: int, basis: str = "z", resets=True, is_final=True):
        """Creates the circuits corresponding to logical basis states.

        Creates the circuits corresponding to logical basis states encoded
        using a rotated surface code.

        Args:
            d (int): Number of code qubits (and hence repetitions) used.
            T (int): Number of rounds of ancilla-assisted syndrome measurement.
            basis (str): Basis used to initialize qubit.
            resets (bool): Whether to include a reset gate after mid-circuit measurements.


        Additional information:
            No measurements are added to the circuit if `T=0`. Otherwise
            `T` rounds are added, followed by measurement of the code
            qubits (corresponding to a logical measurement and final
            syndrome measurement round).
        """
        super().__init__()

        self.d = d
        self.n = d**2
        self.T = 0
        self.basis = basis
        self._resets = resets

        # rotated surface codes tile evenly only for odd d; d=2 is supported as
        # a special case (a valid error-detecting code with an unbalanced
        # Z/X plaquette split)
        if d < 2:
            raise ValueError("Surface code distance must be at least 2.")
        if d % 2 == 0 and d != 2:
            raise NotImplementedError(
                "Even code distances are only supported for d=2, the smallest "
                "even rotated surface code. Larger even d is not implemented."
            )

        # get layout of plaquettes
        self.zplaqs, self.xplaqs, self._zplaq_coords, self._xplaq_coords = self._get_plaquettes()

        self._logicals = {"x": [], "z": []}
        # X logicals for left and right sides
        self._logicals["x"].append([j * self.d for j in range(self.d)])
        self._logicals["x"].append([(j + 1) * self.d - 1 for j in range(self.d)])
        # Z logicals for top and bottom rows
        self._logicals["z"].append(list(range(self.d)))
        self._logicals["z"].append([self.d**2 - 1 - j for j in range(self.d)])

        # set gauge and stabilizer info
        self.x_gauge_ops = [[q for q in plaq if q is not None] for plaq in self.xplaqs]
        self.x_stabilizer_ops = self.x_gauge_ops
        self.x_logical = [self._logicals["x"][0]]
        self.x_boundary = [self._logicals["x"][0] + self._logicals["x"][1]]
        self.z_gauge_ops = [[q for q in plaq if q is not None] for plaq in self.zplaqs]
        self.z_stabilizer_ops = self.z_gauge_ops
        self.z_logical = [self._logicals["z"][0]]
        self.z_boundary = [self._logicals["z"][0] + self._logicals["z"][1]]

        # quantum registers (the number of Z and X plaquettes can differ, e.g. d=2)
        self._num_z = len(self.zplaqs)
        self._num_x = len(self.xplaqs)
        self.code_qubit = QuantumRegister(d**2, "code_qubit")
        self.zplaq_qubit = QuantumRegister(self._num_z, "zplaq_qubit")
        self.xplaq_qubit = QuantumRegister(self._num_x, "xplaq_qubit")
        self.qubit_registers = [self.code_qubit, self.zplaq_qubit, self.xplaq_qubit]

        # classical registers
        self.xplaq_bits = []
        self.zplaq_bits = []
        self.code_bit = ClassicalRegister(d**2, "code_bit")

        # create the circuits
        self.circuit = {}
        for log in ["0", "1"]:
            self.circuit[log] = QuantumCircuit(
                self.code_qubit, self.zplaq_qubit, self.xplaq_qubit, name=log
            )
        self.base = "0"

        # apply initial logical paulis for encoded states
        self._preparation()

        # add the gates required for syndrome measurements
        for _ in range(T - 1):
            self.syndrome_measurement()
        if T != 0 and is_final:
            self.syndrome_measurement(final=True)
            self.readout()


    def _get_plaquettes(self):
        """
        Returns `zplaqs` and `xplaqs`, which are lists of the Z and X type
        stabilizers. Each plaquettes is specified as a list of four qubits,
        in the order in which entangling gates are applied.
        """

        d = self.d

        if d == 2:
            # Canonical distance-2 rotated code (as in stim's rotated d=2): a
            # single Z stabiliser on all four data qubits and X stabilisers on
            # each row. Data layout is row-major:
            #   q0 q1   (top)
            #   q2 q3   (bottom)
            # Every single-qubit error anticommutes with at least one
            # stabiliser, so the code detects (but does not correct) errors.
            zplaqs = [[0, 1, 2, 3]]
            xplaqs = [[0, 1, None, None], [None, None, 2, 3]]
            zplaq_coords = [(0, 0)]
            xplaq_coords = [(0, -1), (0, 1)]
            return zplaqs, xplaqs, zplaq_coords, xplaq_coords

        zplaqs = []
        xplaqs = []
        zplaq_coords = []
        xplaq_coords = []
        for y in range(-1, d):
            for x in range(-1, d):
                bulk = x in range(d - 1) and y in range(d - 1)
                ztab = (x == -1 and y % 2 == 0) or (x == d - 1 and y % 2 == 1)
                xtab = (y == -1 and x % 2 == 1) or (y == d - 1 and x % 2 == 0)

                if (x in range(d - 1) or y in range(d - 1)) and (bulk or ztab or xtab):
                    plaq = []
                    for dy in range(2):
                        for dx in range(2):
                            if x + dx in range(d) and y + dy in range(d):
                                plaq.append(x + dx + d * (y + dy))
                            else:
                                plaq.append(None)

                    if (x + y) % 2 == 0:
                        xplaqs.append([plaq[0], plaq[1], plaq[2], plaq[3]])
                        xplaq_coords.append((x, y))
                    else:
                        zplaqs.append([plaq[0], plaq[2], plaq[1], plaq[3]])
                        zplaq_coords.append((x, y))

        return zplaqs, xplaqs, zplaq_coords, xplaq_coords

    def _preparation(self):
        """
        Prepares logical bit states by applying an x to the circuit that will
        encode a 1.
        """
        if self.basis == "z":
            self.x(["1"])
        else:
            for qc in self.circuit.values():
                qc.h(self.code_qubit)
            self.z(["1"])

    def get_circuit_list(self):
        """
        Returns:
            circuit_list: self.circuit as a list, with
            circuit_list[0] = circuit['0']
            circuit_list[1] = circuit['1']
        """
        circuit_list = [self.circuit[log] for j, log in enumerate(["0", "1"])]
        return circuit_list

    def x(self, logs=("0", "1"), barrier=False):
        """Applies a logical x to the circuits for the given logical values.

        Args:
            logs (list or tuple): List or tuple of logical values expressed as
                strings.
            barrier (bool): Boolean denoting whether to include a barrier at
                the end.
        """
        for log in logs:
            for j in range(self.d):
                self.circuit[log].x(self.code_qubit[j * self.d])
            if barrier:
                self.circuit[log].barrier()

    def z(self, logs=("0", "1"), barrier=False):
        """Applies a logical z to the circuits for the given logical values.

        Args:
            logs (list or tuple): List or tuple of logical values expressed as
                strings.
            barrier (bool): Boolean denoting whether to include a barrier at
                the end.
        """
        for log in logs:
            for j in range(self.d):
                self.circuit[log].z(self.code_qubit[j])
            if barrier:
                self.circuit[log].barrier()

    def syndrome_measurement(self, final=False, barrier=False):
        """Application of a syndrome measurement round.

        Args:
            final (bool): Whether to disregard the reset (if applicable) due to this
                being the final syndrome measurement round.
            barrier (bool): Boolean denoting whether to include a barrier at the end.

        """

        zplaqs, xplaqs = self.zplaqs, self.xplaqs

        # classical registers for this round
        self.zplaq_bits.append(
            ClassicalRegister(self._num_z, "round_" + str(self.T) + "_zplaq_bit")
        )
        self.xplaq_bits.append(
            ClassicalRegister(self._num_x, "round_" + str(self.T) + "_xplaq_bit")
        )

        for log in ["0", "1"]:
            self.circuit[log].add_register(self.zplaq_bits[-1])
            self.circuit[log].add_register(self.xplaq_bits[-1])

            self.circuit[log].h(self.xplaq_qubit)

            for j in range(4):
                for p, plaq in enumerate(zplaqs):
                    c = plaq[j]
                    if c is not None:
                        self.circuit[log].cx(self.code_qubit[c], self.zplaq_qubit[p])
                for p, plaq in enumerate(xplaqs):
                    c = plaq[j]
                    if c is not None:
                        self.circuit[log].cx(self.xplaq_qubit[p], self.code_qubit[c])

            self.circuit[log].h(self.xplaq_qubit)

            for j in range(self._num_x):
                self.circuit[log].measure(self.xplaq_qubit[j], self.xplaq_bits[self.T][j])
                if self._resets and not final:
                    self.circuit[log].reset(self.xplaq_qubit[j])
            for j in range(self._num_z):
                self.circuit[log].measure(self.zplaq_qubit[j], self.zplaq_bits[self.T][j])
                if self._resets and not final:
                    self.circuit[log].reset(self.zplaq_qubit[j])

            if barrier:
                self.circuit[log].barrier()

        self.T += 1

    def readout(self):
        """
        Readout of all code qubits, which corresponds to a logical measurement
        as well as allowing for a measurement of the syndrome to be inferred.
        """

        for log in ["0", "1"]:
            if self.basis == "x":
                self.circuit[log].h(self.code_qubit)
            self.circuit[log].add_register(self.code_bit)
            self.circuit[log].measure(self.code_qubit, self.code_bit)

    def _string2changes(self, string):
        basis = self.basis

        # final syndrome for plaquettes deduced from final code qubit readout
        final_readout = string.split(" ")[0][::-1]
        if basis == "z":
            plaqs = self.zplaqs
        else:
            plaqs = self.xplaqs
        full_syndrome = ""
        for plaq in plaqs:
            parity = 0
            for q in plaq:
                if q is not None:
                    parity += int(final_readout[q])
            full_syndrome = str(parity % 2) + full_syndrome

        # results from all other plaquette syndrome measurements then added
        if basis == "z":
            full_syndrome = full_syndrome + " " + " ".join(string.split(" ")[2::2])
        else:
            full_syndrome = full_syndrome + " " + " ".join(string.split(" ")[1::2])

        # changes between one syndrome and the next then calculated
        syndrome_list = full_syndrome.split(" ")
        height = len(syndrome_list)
        width = len(syndrome_list[0])
        syndrome_changes = ""
        for t in range(height):
            for j in range(width):
                if self._resets:
                    if t == 0:
                        change = syndrome_list[-1][j] != "0"
                    else:
                        change = syndrome_list[-t][j] != syndrome_list[-t - 1][j]
                    syndrome_changes += "0" * (not change) + "1" * change
                else:
                    if t <= 1:
                        if t != self.T:
                            change = syndrome_list[-t - 1][j] != "0"
                        else:
                            change = syndrome_list[-t - 1][j] != syndrome_list[-t][j]
                    elif t == self.T:
                        last3 = ""
                        for dt in range(3):
                            last3 += syndrome_list[-t - 1 + dt][j]
                        change = last3.count("1") % 2 == 1
                    else:
                        change = syndrome_list[-t - 1][j] != syndrome_list[-t + 1][j]
                    syndrome_changes += "0" * (not change) + "1" * change
            syndrome_changes += " "
        syndrome_changes = syndrome_changes[0:-1]

        if basis != self.basis:
            # trim the noisy nonsense (first and last rounds)
            syndrome_changes = " ".join(syndrome_changes.split(" ")[1:-1])

        return syndrome_changes

    def string2raw_logicals(self, string):
        """
        Extracts raw logicals from output string.
        Args:
            string (string): Results string from which to extract logicals
        Returns:
            list: Raw values for logical operators that correspond to nodes.
        """
        final_readout = string.split(" ")[0][::-1]
        # get logical readout
        # (though it's called Z, it actually depends on the basis)
        Z = [0, 0]
        for j in range(self.d):
            if self.basis == "z":
                # evaluated using top row
                Z[0] = (Z[0] + int(final_readout[j])) % 2
                # evaluated using bottom row
                Z[1] = (Z[1] + int(final_readout[self.d**2 - 1 - j])) % 2
            else:
                # evaluated using left side
                Z[0] = (Z[0] + int(final_readout[j * self.d])) % 2
                # evaluated using right side
                Z[1] = (Z[1] + int(final_readout[(j + 1) * self.d - 1])) % 2
        return [str(Z[0]), str(Z[1])]

    def _process_string(self, string):
        # get logical readout
        measured_Z = self.string2raw_logicals(string)

        # then get syndrome changes
        syndrome_changes = self._string2changes(string)

        # the space separated string of syndrome changes then gets a
        # double space separated logical value on the end
        new_string = " ".join(measured_Z) + "  " + syndrome_changes

        return new_string

    def _separate_string(self, string):
        separated_string = []
        for syndrome_type_string in string.split("  "):
            separated_string.append(syndrome_type_string.split(" "))
        return separated_string

    def string2nodes(self, string, **kwargs):
        """Convert output string from circuits into a set of nodes.

        Args:
            string (string): Results string to convert.
            kwargs (dict): Additional keyword arguments. See below:

        kwargs:
            logical (str): Logical value whose results are used ('0' as default).
            all_logicals (bool): Whether to include logical nodes
            irrespective of value. (False as default).

        Returns:
            dict: List of nodes corresponding to to the non-trivial
                elements in the string.

        Additional information:
            Strings are read right to left, but lists*
            are read left to right. So, we have some ugly indexing
            code whenever we're dealing with both strings and lists.
        """

        all_logicals = kwargs.get("all_logicals")
        logical = kwargs.get("logical")
        if logical is None:
            logical = "0"

        string = self._process_string(string)
        separated_string = self._separate_string(string)
        nodes = []

        # boundary nodes
        boundary = separated_string[0]  # [<last_elem>, <init_elem>]
        for bqec_index, belement in enumerate(boundary[::-1]):
            if all_logicals or belement != logical:
                node = DecodingGraphNode(
                    is_logical=True,
                    is_boundary=True,
                    qubits=self._logicals[self.basis][-bqec_index - 1],
                    index=1 - bqec_index,
                )
                nodes.append(node)

        # bulk nodes
        for syn_type in range(1, len(separated_string)):
            for syn_round in range(len(separated_string[syn_type])):
                elements = separated_string[syn_type][syn_round]
                for qec_index, element in enumerate(elements[::-1]):
                    if element == "1":
                        if self.basis == "x":
                            qubits = self.x_stabilizer_ops[qec_index]
                        else:
                            qubits = self.z_stabilizer_ops[qec_index]
                        node = DecodingGraphNode(time=syn_round, qubits=qubits, index=qec_index)
                        nodes.append(node)
        return nodes

    def check_nodes(self, nodes, ignore_extras=False, minimal=False):
        """
        Determines whether a given set of nodes are neutral. If so, also
        determines any additional logical readout qubits that would be
        flipped by the errors creating such a cluster and how many errors
        would be required to make the cluster.
        Args:
            nodes (list): List of nodes, of the type produced by `string2nodes`.
            ignore_extras (bool): If `True`, undeeded logical nodes are
            ignored.
            minimal (bool): Whether output should only reflect the minimal error
            case.
        Returns:
            neutral (bool): Whether the nodes independently correspond to a valid
            set of errors.
            flipped_logical_nodes (list): List of qubits nodes for logical
            operators that are flipped by the errors, that were not included
            in the original nodes.
            num_errors (int): Minimum number of errors required to create nodes.
        """

        bulk_nodes = [node for node in nodes if not node.is_logical]
        logical_nodes = [node for node in nodes if node.is_logical]
        given_logicals = set(node.index for node in logical_nodes)

        if self.basis == "z":
            coords = self._zplaq_coords
        else:
            coords = self._xplaq_coords

        if (len(bulk_nodes) % 2) == 0:
            if (len(logical_nodes) % 2) == 0 or ignore_extras:
                neutral = True
                flipped_logicals = set()
                # estimate num_errors from size
                if bulk_nodes:
                    xs = []
                    ys = []
                    for node in bulk_nodes:
                        x, y = coords[node.index]
                        xs.append(x)
                        ys.append(y)
                    dx = max(xs) - min(xs)
                    dy = max(ys) - min(ys)
                    num_errors = dx + dy
                    if dx > 0 and dy > 0:
                        num_errors -= 1
                else:
                    num_errors = 0
            else:
                neutral = False
                flipped_logicals = set()
                num_errors = 0
        else:
            # find nearest boundary
            num_errors = (self.d - 1) / 2
            for node in bulk_nodes:
                x, y = coords[node.index]
                if self.basis == "z":
                    p = y
                else:
                    p = x
                num_errors = min(num_errors, p + 1, self.d - p)
            flipped_logicals = {1 - int(p < (self.d - 1) / 2)}

        # if unneeded logical zs are given, cluster is not neutral
        # (unless this is ignored)
        if (not ignore_extras) and given_logicals.difference(flipped_logicals):
            neutral = False
        # otherwise, report only needed logicals that aren't given
        else:
            neutral = True
            flipped_logicals = flipped_logicals.difference(given_logicals)

        # get the required boundary nodes
        flipped_logical_nodes = []
        for elem in flipped_logicals:
            node = DecodingGraphNode(
                is_logical=True,
                is_boundary=True,
                qubits=self._logicals[self.basis][elem],
                index=elem,
            )
            flipped_logical_nodes.append(node)

        return neutral, flipped_logical_nodes, num_errors

    def is_cluster_neutral(self, nodes):
        """
        Determines whether or not the cluster is neutral, meaning that one or more
        errors could have caused the set of nodes (syndrome changes) passed
        to the method.
        Args:
            nodes (dictionary in the form of the return value of string2nodes)
        """
        return not bool(len(nodes) % 2)



class RoughMergeCircuit(SurfaceCodeCircuit):
    """Two distance-d rotated surface codes side-by-side, undergoing a rough
    (X_L X_L) merge. The merged circuit combines the left and right code
    circuits and adds a row of `d` X-parity checks across the seam between
    them; the XOR of the seam outcomes equals X_L(left) * X_L(right).
    """

    def __init__(self, d, T, basis="z", resets=True, is_final=True, flip=None):
        # T=0 keeps the base init from building an unused standalone code; the
        # left/right codes below are the real ones carrying the syndrome rounds.
        super().__init__(d, T=0, basis=basis, resets=resets, is_final=False)

        self.left_code = SurfaceCodeCircuit(d=d, T=T, basis=basis, resets=resets, is_final=False)
        self.right_code = SurfaceCodeCircuit(d=d, T=T, basis=basis, resets=resets, is_final=False)

        if flip == "z":
            self.left_code.z(["0", "1"])

        elif flip == "x":
            self.left_code.x(["0", "1"])

        # the merge seam: right column of the left code faces the left column
        # of the right code, row by row (both are X-logical boundaries)
        self.merge_ancillas = d
        self.merge_ancilla = QuantumRegister(d, "merge_qubit")
        self.merge_pairs = [((j + 1) * d - 1, j * d) for j in range(d)]

        self.d = d
        self.n = 2 * d**2 + self.merge_ancillas
        self.T = self.left_code.T
        self.basis = basis
        self._resets = resets

        self.circuit = {}
        for log in ("0", "1"):
            self.circuit[log] = self._build_merged_circuit(log)
        self.base = "0"

        if is_final:
            self.readout()

    def readout(self):
        """Final measurement of both codes' data qubits, corresponding to a
        logical measurement of each code (and allowing a final syndrome to be
        inferred). As with the single code, the data qubits are measured in X
        when the codes were prepared in the X basis.
        """
        for log in ("0", "1"):
            qc = self.circuit[log]
            reg_l = next(r for r in qc.qregs if r.name == "code_qubit")
            reg_r = next(r for r in qc.qregs if r.name == "right_code_qubit")

            creg_names = {r.name for r in qc.cregs}
            if "code_bit" in creg_names:
                bits_l = next(r for r in qc.cregs if r.name == "code_bit")
            else:
                bits_l = ClassicalRegister(self.d**2, "code_bit")
                qc.add_register(bits_l)
            if "right_code_bit" in creg_names:
                bits_r = next(r for r in qc.cregs if r.name == "right_code_bit")
            else:
                bits_r = ClassicalRegister(self.d**2, "right_code_bit")
                qc.add_register(bits_r)

            if self.basis == "x":
                qc.h(reg_l)
                qc.h(reg_r)
            qc.measure(reg_l, bits_l)
            qc.measure(reg_r, bits_r)

    def _build_merged_circuit(self, log):
        """Combine the left and right code circuits into a single circuit,
        then append the X-parity merge round(s).

        The left code's registers are reused as-is; the right code's registers
        are copied under a "right_" prefix so register names do not collide.
        """
        qc_l = self.left_code.circuit[log]
        qc_r = self.right_code.circuit[log]

        right_qregs = [QuantumRegister(r.size, "right_" + r.name) for r in qc_r.qregs]
        right_cregs = [ClassicalRegister(r.size, "right_" + r.name) for r in qc_r.cregs]

        combined = QuantumCircuit(
            *qc_l.qregs, *qc_l.cregs,
            *right_qregs, *right_cregs,
            self.merge_ancilla,
            name="merge_" + log,
        )

        # qc_l as base, then qc_r's qubits and gates appended onto it
        left_qubits = [q for reg in qc_l.qregs for q in reg]
        left_clbits = [c for reg in qc_l.cregs for c in reg]
        right_qubits = [q for reg in right_qregs for q in reg]
        right_clbits = [c for reg in right_cregs for c in reg]
        combined.compose(qc_l, qubits=left_qubits, clbits=left_clbits, inplace=True)
        combined.compose(qc_r, qubits=right_qubits, clbits=right_clbits, inplace=True)

        # data qubits of each code within the combined circuit (right registers
        # follow all of the left code's registers, including its syndrome
        # ancillas)
        code_l = combined.qubits[0:self.d**2]
        code_r = combined.qubits[qc_l.num_qubits:qc_l.num_qubits + self.d**2]

        # one seam round per syndrome round the codes ran (at least one)
        n_merge = self.left_code.T or 1
        for t in range(n_merge):
            self._merge_round(combined, code_l, code_r, t)

        return combined

    def _merge_round(self, merged, code_l, code_r, t):
        """One X-parity round across the seam, mirroring an X-plaquette
        measurement: |0>, H, CNOT to each facing data qubit, H, measure.
        """
        creg = ClassicalRegister(self.merge_ancillas,
                                 "round_" + str(t) + "_merge_bit")
        merged.add_register(creg)

        merged.h(self.merge_ancilla)
        for k, (left_idx, right_idx) in enumerate(self.merge_pairs):
            merged.cx(self.merge_ancilla[k], code_l[left_idx])
            merged.cx(self.merge_ancilla[k], code_r[right_idx])
        merged.h(self.merge_ancilla)

        for k in range(self.merge_ancillas):
            merged.measure(self.merge_ancilla[k], creg[k])
            if self._resets:
                merged.reset(self.merge_ancilla[k])


class SmoothMergeCircuit(SurfaceCodeCircuit):
    """Two distance-d rotated surface codes stacked vertically, undergoing a
    smooth (Z_L Z_L) merge. The merged circuit combines the two code circuits
    and adds a row of `d` Z-parity checks along the seam between the bottom
    row of the upper code and the top row of the lower one; the XOR of the
    seam outcomes equals Z_L(upper) * Z_L(lower).
    """

    def __init__(self, d, T, basis="z", resets=True, is_final=True, flip=None):
        # T=0 keeps the base init from building an unused standalone code; the
        # upper/lower codes below are the real ones carrying the syndrome rounds.
        super().__init__(d, T=0, basis=basis, resets=resets, is_final=False)

        self.upper_code = SurfaceCodeCircuit(d=d, T=T, basis=basis, resets=resets, is_final=False)
        self.lower_code = SurfaceCodeCircuit(d=d, T=T, basis=basis, resets=resets, is_final=False)

        if flip == "z":
            self.upper_code.z(["0", "1"])
        
        elif flip == "x":
            self.upper_code.x(["0", "1"])

        # the merge seam: bottom row of the upper code faces the top row of
        # the lower code, column by column (both are Z-logical boundaries)
        self.merge_ancillas = d
        self.merge_ancilla = QuantumRegister(d, "merge_qubit")
        self.merge_pairs = [(d * (d - 1) + j, j) for j in range(d)]

        self.d = d
        self.n = 2 * d**2 + self.merge_ancillas
        self.T = self.upper_code.T
        self.basis = basis
        self._resets = resets

        self.circuit = {}
        for log in ("0", "1"):
            self.circuit[log] = self._build_merged_circuit(log)
        self.base = "0"

        if is_final:
            self.readout()

    def readout(self):
        """Final measurement of both codes' data qubits, corresponding to a
        logical measurement of each code. As with the single code, the data
        qubits are measured in X when the codes were prepared in the X basis.
        """
        for log in ("0", "1"):
            qc = self.circuit[log]
            reg_u = next(r for r in qc.qregs if r.name == "code_qubit")
            reg_l = next(r for r in qc.qregs if r.name == "right_code_qubit")

            creg_names = {r.name for r in qc.cregs}
            if "code_bit" in creg_names:
                bits_u = next(r for r in qc.cregs if r.name == "code_bit")
            else:
                bits_u = ClassicalRegister(self.d**2, "code_bit")
                qc.add_register(bits_u)
            if "right_code_bit" in creg_names:
                bits_l = next(r for r in qc.cregs if r.name == "right_code_bit")
            else:
                bits_l = ClassicalRegister(self.d**2, "right_code_bit")
                qc.add_register(bits_l)

            if self.basis == "x":
                qc.h(reg_u)
                qc.h(reg_l)
            qc.measure(reg_u, bits_u)
            qc.measure(reg_l, bits_l)

    def _build_merged_circuit(self, log):
        """Combine the upper and lower code circuits into a single circuit,
        then append the Z-parity merge round(s).

        The upper code's registers are reused as-is; the lower code's
        registers are copied under a "right_" prefix so register names do not
        collide.
        """
        qc_u = self.upper_code.circuit[log]
        qc_l = self.lower_code.circuit[log]

        right_qregs = [QuantumRegister(r.size, "right_" + r.name) for r in qc_l.qregs]
        right_cregs = [ClassicalRegister(r.size, "right_" + r.name) for r in qc_l.cregs]

        combined = QuantumCircuit(
            *qc_u.qregs, *qc_u.cregs,
            *right_qregs, *right_cregs,
            self.merge_ancilla,
            name="merge_" + log,
        )

        # qc_u as base, then qc_l's qubits and gates appended onto it
        upper_qubits = [q for reg in qc_u.qregs for q in reg]
        upper_clbits = [c for reg in qc_u.cregs for c in reg]
        lower_qubits = [q for reg in right_qregs for q in reg]
        lower_clbits = [c for reg in right_cregs for c in reg]
        combined.compose(qc_u, qubits=upper_qubits, clbits=upper_clbits, inplace=True)
        combined.compose(qc_l, qubits=lower_qubits, clbits=lower_clbits, inplace=True)

        # data qubits of each code within the combined circuit (lower
        # registers follow all of the upper code's registers, including its
        # syndrome ancillas)
        code_u = combined.qubits[0:self.d**2]
        code_l = combined.qubits[qc_u.num_qubits:qc_u.num_qubits + self.d**2]

        # one seam round per syndrome round the codes ran (at least one)
        n_merge = self.upper_code.T or 1
        for t in range(n_merge):
            self._merge_round(combined, code_u, code_l, t)

        return combined

    def _merge_round(self, merged, code_u, code_l, t):
        """One Z-parity round across the seam, mirroring a Z-plaquette
        measurement: |0>, CNOT from each facing data qubit to the ancilla,
        measure.
        """
        creg = ClassicalRegister(self.merge_ancillas,
                                 "round_" + str(t) + "_merge_bit")
        merged.add_register(creg)

        for k, (upper_idx, lower_idx) in enumerate(self.merge_pairs):
            merged.cx(code_u[upper_idx], self.merge_ancilla[k])
            merged.cx(code_l[lower_idx], self.merge_ancilla[k])

        for k in range(self.merge_ancillas):
            merged.measure(self.merge_ancilla[k], creg[k])
            if self._resets:
                merged.reset(self.merge_ancilla[k])




