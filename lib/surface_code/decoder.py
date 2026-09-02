# -*- coding: utf-8 -*-
"""
Minimum-weight perfect matching (MWPM) decoder built on PyMatching.

Decodes the syndrome changes produced by ``SurfaceCodeCircuit.string2nodes``.
For a given code (built once) a space-time ``pymatching.Matching`` graph is
created whose vertices are every (stabiliser, round) site that can fire, with

  - space edges between stabilisers sharing a data qubit (data errors),
  - time edges between consecutive rounds of the same stabiliser (measurement
    errors),
  - boundary edges to the two boundaries of the measured type, tagged with the
    logical index of that boundary (fault ids ``0`` / ``1``).

The ``string2nodes`` output for a shot is turned into a syndrome bit vector
(the bulk, non-logical nodes) and decoded. The correction for boundary row
``i`` is the parity of matched boundary edges with fault id ``i``; XOR-ing it
with the measured logical readout of that row (``string2raw_logicals``) gives
the MWPM-corrected logical value.
"""

import numpy as np
import pymatching


class MWPM_Decoder:
    """Minimum weight perfect matching decoder for a surface code."""

    def __init__(self, code, weights=None):
        """Build the space-time matching graph for ``code``.

        Args:
            code (SurfaceCodeCircuit): Code to decode. ``code.basis`` selects
                which stabiliser type is decoded ('z' plaquettes against the
                top/bottom boundaries, or 'x' plaquettes against the left/right
                boundaries).
            weights (dict): Optional edge weights keyed by mechanism type,
                e.g. ``{'data': 1.0, 'measurement': 1.0}``. If ``None`` every
                single-qubit error mechanism gets weight 1, so the decoder
                minimises the number of errors (no physical error model needed).
        """
        self.code = code
        if code.T == 0:
            raise ValueError("MWPM decoding requires at least one syndrome "
                             "measurement round (T > 0).")

        if code.basis == "z":
            self.ops = code.z_stabilizer_ops
            self.boundaries = code._logicals["z"]
        else:
            self.ops = code.x_stabilizer_ops
            self.boundaries = code._logicals["x"]

        self.num_syndromes = len(self.ops)
        # syndrome changes are reported for one extra layer: the difference
        # between the last ancilla round and the final (readout-inferred) round
        self.layers = code.T + 1
        self.num_detectors = self.num_syndromes * self.layers

        if weights is None:
            weights = {}
        self._w_data = weights.get("data", 1.0)
        self._w_meas = weights.get("measurement", 1.0)

        self.matching = self._build_matching()

    def _vertex(self, p, t):
        return p * self.layers + t

    def _build_matching(self):
        opsets = [set(support) for support in self.ops]
        boundary_sets = [set(row) for row in self.boundaries]

        # stabilisers are adjacent when they share a data qubit; a single error
        # on that qubit flips both
        neighbours = [set() for _ in range(self.num_syndromes)]
        for p in range(self.num_syndromes):
            for q in range(p + 1, self.num_syndromes):
                if opsets[p] & opsets[q]:
                    neighbours[p].add(q)
                    neighbours[q].add(p)

        matching = pymatching.Matching()
        matching.ensure_num_fault_ids(2)
        for p in range(self.num_syndromes):
            for q in neighbours[p]:
                if q > p:
                    for t in range(self.layers):
                        matching.add_edge(
                            self._vertex(p, t), self._vertex(q, t),
                            weight=self._w_data)
            for t in range(self.layers - 1):
                matching.add_edge(
                    self._vertex(p, t), self._vertex(p, t + 1),
                    weight=self._w_meas)
            for boundary in range(2):
                if opsets[p] & boundary_sets[boundary]:
                    for t in range(self.layers):
                        matching.add_boundary_edge(
                            self._vertex(p, t),
                            weight=self._w_data, fault_ids={boundary})
        return matching

    def nodes2syndrome(self, nodes):
        """Convert a ``string2nodes`` list into a pymatching syndrome vector.

        Only the bulk (non-logical) nodes are syndrome events. The logical
        boundary nodes are readout discrepancies and are accounted for through
        the measured logical values passed to ``decode``.

        Args:
            nodes (list): Output of ``SurfaceCodeCircuit.string2nodes``.
        Returns:
            numpy.ndarray: Syndrome bit vector of length ``num_detectors``.
        """
        syndrome = np.zeros(self.num_detectors, dtype=np.uint8)
        for node in nodes:
            if node.is_logical:
                continue
            syndrome[self._vertex(node.index, node.time)] = 1
        return syndrome

    def predict(self, nodes):
        """Decode a syndrome and return the predicted boundary flips.

        Args:
            nodes (list): Output of ``SurfaceCodeCircuit.string2nodes``.
        Returns:
            list: ``[p0, p1]`` where ``pi`` is 1 if the matching inferred that
            an error chain reached boundary ``i``.
        """
        predicted = self.matching.decode(self.nodes2syndrome(nodes))
        return [int(predicted[0]), int(predicted[1])]

    def decode(self, nodes, raw_logicals):
        """Return MWPM-corrected logical values for a shot.

        Args:
            nodes (list): Output of ``SurfaceCodeCircuit.string2nodes``.
            raw_logicals (list): The two measured logical readouts, in the same
                order as ``SurfaceCodeCircuit.string2raw_logicals`` returns
                them (i.e. the raw values before correction).
        Returns:
            list: ``[c0, c1]``, the corrected logical values for the two
            boundaries. For a shot with no uncorrectable logical error these
            equal the prepared logical value.
        """
        predicted = self.predict(nodes)
        return [int(raw_logicals[i]) ^ predicted[i] for i in range(2)]

    def decode_string(self, string, logical="0"):
        """Decode a raw result string (as returned by AerSimulator counts).

        Args:
            string (str): Result string for a single shot.
            logical (str): Prepared logical value ('0' or '1'); only used to
                select the logical boundary nodes from ``string2nodes``, which
                do not enter the syndrome.
        Returns:
            list: ``[c0, c1]`` corrected logical values.
        """
        nodes = self.code.string2nodes(string, logical=logical)
        raw = self.code.string2raw_logicals(string)
        return self.decode(nodes, raw)
