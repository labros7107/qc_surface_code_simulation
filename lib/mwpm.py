"""
Surface code logical measurement via MWPM (PyMatching).

Layout — 5x5 unrotated surface code:
  Data qubits   : 0,2,4,6,8,10,12,14,16,18,20,22,24  (even coords)
  Ancilla qubits: 1,3,5,7,9,11,13,15,17,19,21,23      (odd coords)

Logical Z-observable (OBSERVABLE_INCLUDE): rightmost column x=4
  → data qubits 4, 14, 24  (rec[-13], rec[-10], rec[-7], rec[-4], rec[-1])

Usage
-----
    outcomes = measure_logical(circuit, num_shots=1000)
    print(f"Logical error rate: {outcomes.mean():.4f}")
"""

import stim
import pymatching
import numpy as np


# # ---------------------------------------------------------------------------
# # Circuit
# # ---------------------------------------------------------------------------

SURFACE_CODE_CIRCUIT = stim.Circuit("""
QUBIT_COORDS(0, 0) 0
QUBIT_COORDS(1, 0) 1
QUBIT_COORDS(2, 0) 2
QUBIT_COORDS(3, 0) 3
QUBIT_COORDS(4, 0) 4
QUBIT_COORDS(0, 1) 5
QUBIT_COORDS(1, 1) 6
QUBIT_COORDS(2, 1) 7
QUBIT_COORDS(3, 1) 8
QUBIT_COORDS(4, 1) 9
QUBIT_COORDS(0, 2) 10
QUBIT_COORDS(1, 2) 11
QUBIT_COORDS(2, 2) 12
QUBIT_COORDS(3, 2) 13
QUBIT_COORDS(4, 2) 14
QUBIT_COORDS(0, 3) 15
QUBIT_COORDS(1, 3) 16
QUBIT_COORDS(2, 3) 17
QUBIT_COORDS(3, 3) 18
QUBIT_COORDS(4, 3) 19
QUBIT_COORDS(0, 4) 20
QUBIT_COORDS(1, 4) 21
QUBIT_COORDS(2, 4) 22
QUBIT_COORDS(3, 4) 23
QUBIT_COORDS(4, 4) 24
R 0 2 4 6 8 10 12 14 16 18 20 22 24 1 3 5 7 9 11 13 15 17 19 21 23
TICK
H 1 3 11 13 21 23
TICK
CX 1 2 11 12 21 22 3 4 13 14 23 24 6 5 16 15 8 7 18 17
TICK
CX 1 6 11 16 3 8 13 18 10 5 20 15 12 7 22 17 14 9 24 19
TICK
CX 11 6 21 16 13 8 23 18 0 5 10 15 2 7 12 17 4 9 14 19
TICK
CX 1 0 11 10 21 20 3 2 13 12 23 22 6 7 16 17 8 9 18 19
TICK
H 1 3 11 13 21 23
TICK
MR 1 3 5 7 9 11 13 15 17 19 21 23
DETECTOR(0, 1, 0) rec[-10]
DETECTOR(0, 3, 0) rec[-5]
DETECTOR(2, 1, 0) rec[-9]
DETECTOR(2, 3, 0) rec[-4]
DETECTOR(4, 1, 0) rec[-8]
DETECTOR(4, 3, 0) rec[-3]
TICK
H 1 3 11 13 21 23
TICK
CX 1 2 11 12 21 22 3 4 13 14 23 24 6 5 16 15 8 7 18 17
TICK
CX 1 6 11 16 3 8 13 18 10 5 20 15 12 7 22 17 14 9 24 19
TICK
CX 11 6 21 16 13 8 23 18 0 5 10 15 2 7 12 17 4 9 14 19
TICK
CX 1 0 11 10 21 20 3 2 13 12 23 22 6 7 16 17 8 9 18 19
TICK
H 1 3 11 13 21 23
TICK
MR 1 3 5 7 9 11 13 15 17 19 21 23
SHIFT_COORDS(0, 0, 1)
DETECTOR(1, 0, 0) rec[-12] rec[-24]
DETECTOR(3, 0, 0) rec[-11] rec[-23]
DETECTOR(0, 1, 0) rec[-10] rec[-22]
DETECTOR(2, 1, 0) rec[-9] rec[-21]
DETECTOR(4, 1, 0) rec[-8] rec[-20]
DETECTOR(1, 2, 0) rec[-7] rec[-19]
DETECTOR(3, 2, 0) rec[-6] rec[-18]
DETECTOR(0, 3, 0) rec[-5] rec[-17]
DETECTOR(2, 3, 0) rec[-4] rec[-16]
DETECTOR(4, 3, 0) rec[-3] rec[-15]
DETECTOR(1, 4, 0) rec[-2] rec[-14]
DETECTOR(3, 4, 0) rec[-1] rec[-13]
TICK
Z_ERROR(1) 0 2 4
TICK
H 1 3 11 13 21 23
TICK
CX 1 2 11 12 21 22 3 4 13 14 23 24 6 5 16 15 8 7 18 17
TICK
CX 1 6 11 16 3 8 13 18 10 5 20 15 12 7 22 17 14 9 24 19
TICK
CX 11 6 21 16 13 8 23 18 0 5 10 15 2 7 12 17 4 9 14 19
TICK
CX 1 0 11 10 21 20 3 2 13 12 23 22 6 7 16 17 8 9 18 19
TICK
H 1 3 11 13 21 23
TICK
MR 1 3 5 7 9 11 13 15 17 19 21 23
SHIFT_COORDS(0, 0, 1)
DETECTOR(1, 0, 0) rec[-12] rec[-24]
DETECTOR(3, 0, 0) rec[-11] rec[-23]
DETECTOR(0, 1, 0) rec[-10] rec[-22]
DETECTOR(2, 1, 0) rec[-9] rec[-21]
DETECTOR(4, 1, 0) rec[-8] rec[-20]
DETECTOR(1, 2, 0) rec[-7] rec[-19]
DETECTOR(3, 2, 0) rec[-6] rec[-18]
DETECTOR(0, 3, 0) rec[-5] rec[-17]
DETECTOR(2, 3, 0) rec[-4] rec[-16]
DETECTOR(4, 3, 0) rec[-3] rec[-15]
DETECTOR(1, 4, 0) rec[-2] rec[-14]
DETECTOR(3, 4, 0) rec[-1] rec[-13]
TICK
H 1 3 11 13 21 23
TICK
CX 1 2 11 12 21 22 3 4 13 14 23 24 6 5 16 15 8 7 18 17
TICK
CX 1 6 11 16 3 8 13 18 10 5 20 15 12 7 22 17 14 9 24 19
TICK
CX 11 6 21 16 13 8 23 18 0 5 10 15 2 7 12 17 4 9 14 19
TICK
CX 1 0 11 10 21 20 3 2 13 12 23 22 6 7 16 17 8 9 18 19
TICK
H 1 3 11 13 21 23
TICK
MR 1 3 5 7 9 11 13 15 17 19 21 23
SHIFT_COORDS(0, 0, 1)
DETECTOR(1, 0, 0) rec[-12] rec[-24]
DETECTOR(3, 0, 0) rec[-11] rec[-23]
DETECTOR(0, 1, 0) rec[-10] rec[-22]
DETECTOR(2, 1, 0) rec[-9] rec[-21]
DETECTOR(4, 1, 0) rec[-8] rec[-20]
DETECTOR(1, 2, 0) rec[-7] rec[-19]
DETECTOR(3, 2, 0) rec[-6] rec[-18]
DETECTOR(0, 3, 0) rec[-5] rec[-17]
DETECTOR(2, 3, 0) rec[-4] rec[-16]
DETECTOR(4, 3, 0) rec[-3] rec[-15]
DETECTOR(1, 4, 0) rec[-2] rec[-14]
DETECTOR(3, 4, 0) rec[-1] rec[-13]
M 0 2 4 6 8 10 12 14 16 18 20 22 24
DETECTOR(0, 1, 1) rec[-8] rec[-10] rec[-13] rec[-23]
DETECTOR(0, 3, 1) rec[-3] rec[-5] rec[-8] rec[-18]
DETECTOR(2, 1, 1) rec[-7] rec[-9] rec[-10] rec[-12] rec[-22]
DETECTOR(2, 3, 1) rec[-2] rec[-4] rec[-5] rec[-7] rec[-17]
DETECTOR(4, 1, 1) rec[-6] rec[-9] rec[-11] rec[-21]
DETECTOR(4, 3, 1) rec[-1] rec[-4] rec[-6] rec[-16]
OBSERVABLE_INCLUDE(0) rec[-1] rec[-4] rec[-7] rec[-10] rec[-13]
""")


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def build_mwpm_decoder(
    circuit: stim.Circuit,
    p_clamp: float = 0.499,
) -> pymatching.Matching:
    """
    Build a PyMatching MWPM decoder from a stim circuit.

    Stim converts the circuit's noise into a Detector Error Model (DEM):
      - Nodes  = detectors (stabiliser measurement outcomes)
      - Edges  = error mechanisms that flip exactly two detectors
      - Weight = -log(p / (1-p)) for each error probability p
      - Boundary edges = errors that flip only one detector (code boundary)

    PyMatching then finds the minimum-weight perfect matching on the set of
    fired detectors, which gives the most likely error chain.

    Parameters
    ----------
    circuit : stim.Circuit
        Must contain DETECTOR and OBSERVABLE_INCLUDE annotations.
    p_clamp : float
        Maximum edge probability. DEM edges with p >= p_clamp are clamped to
        p_clamp before building the matching graph.

        This is necessary when the circuit contains X_ERROR(1) or Z_ERROR(1)
        (deterministic errors): p=1.0 maps to an infinite matching weight
        that PyMatching cannot represent. Clamping to 0.499 keeps the edge
        as the heaviest possible while remaining numerically valid — it still
        encodes "this error is overwhelmingly likely" to the decoder.

    Returns
    -------
    pymatching.Matching
        Ready-to-use MWPM decoder.
    """
    dem = circuit.detector_error_model(decompose_errors=True)

    # Rebuild the DEM with all probabilities clamped below 1.0
    clamped_dem = stim.DetectorErrorModel()
    for inst in dem.flattened():
        if inst.type == "error":
            p = min(inst.args_copy()[0], p_clamp)
            clamped_dem.append(
                stim.DemInstruction("error", [p], inst.targets_copy())
            )
        else:
            clamped_dem.append(inst)

    return pymatching.Matching.from_detector_error_model(clamped_dem)


def measure_logical(
    circuit: stim.Circuit,
    num_shots: int = 1,
    *,
    p_clamp: float = 0.499,
    seed: int | None = None,
) -> np.ndarray:
    """
    Sample the circuit and return MWPM-corrected logical measurement outcomes.

    Pipeline per shot
    -----------------
    1. Stim samples detector syndromes and raw observable parities.
    2. MWPM finds the minimum-weight error chain consistent with the syndrome.
    3. The chain's net effect on the logical observable is the predicted
       correction (0 = no flip needed, 1 = flip needed).
    4. corrected = raw_observable_parity XOR predicted_correction

    If MWPM finds the right chain the XOR cancels the error → result = 0.
    If the error is uncorrectable (weight >= d/2) MWPM picks the wrong chain
    and the XOR doubles it → result = 1 (logical failure flagged).

    Parameters
    ----------
    circuit : stim.Circuit
        Annotated circuit with DETECTOR and OBSERVABLE_INCLUDE.
    num_shots : int
        Number of independent shots to sample and decode.
    p_clamp : float
        Passed to build_mwpm_decoder; clamps p=1 DEM edges.
    seed : int or None
        RNG seed for reproducibility.

    Returns
    -------
    np.ndarray, shape (num_shots,), dtype uint8
        Corrected logical Z outcomes per shot.
        0 → no net logical error after correction.
        1 → logical error detected (uncorrectable by MWPM).
    """
    matcher = build_mwpm_decoder(circuit, p_clamp=p_clamp)

    sampler = circuit.compile_detector_sampler(seed=seed)
    # det_samples : (num_shots, num_detectors)   — syndrome bits
    # obs_samples : (num_shots, num_observables) — raw logical parity bits
    det_samples, obs_samples = sampler.sample(
        shots=num_shots,
        separate_observables=True,
    )

    corrected = np.empty(num_shots, dtype=np.uint8)
    for i in range(num_shots):
        # predicted_flip[j] = 1 if MWPM thinks observable j was flipped
        predicted_flip = matcher.decode(det_samples[i])
        # XOR: raw parity cancels with correction if MWPM is right
        corrected[i] = int(obs_samples[i, 0]) ^ int(predicted_flip[0])

    return corrected


def exec_mwpm(circuit, shots, seed):

    print(f"--- {shots}-shot batch decode ---")
    outcomes = measure_logical(circuit, num_shots=shots, seed=seed)

    sampler = circuit.compile_detector_sampler(seed=seed)
    _, obs_raw = sampler.sample(shots=shots, separate_observables=True)
    raw = obs_raw[:, 0].astype(int).tolist()

    print(f"  Raw observable parities   : {raw}")
    print(f"  Corrected logical outcomes: {outcomes.tolist()}")
    print(f"  Logical error rate        : {outcomes.mean():.2f}")
    print()
    print("  Raw = 1 on all shots  → X error flipped the logical Z.")
    print("  Corrected = 0         → MWPM identified and cancelled it.")


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 55)
    print("  Surface code logical measurement via MWPM")
    print("=" * 55)
    print("""
This circuit injects X_ERROR(1) on data qubits {0, 2, 4} —
the complete top boundary row. This is a weight-3 logical X
error (it anticommutes with Z̄). Without decoding, different
homologous Z-strings give different parities. MWPM reconciles
them by finding the most likely error chain and computing a
net correction to the logical observable.
""")

    # --- 10-shot batch ---
    print("--- 10-shot batch decode ---")
    outcomes = measure_logical(SURFACE_CODE_CIRCUIT, num_shots=10, seed=42)

    sampler = SURFACE_CODE_CIRCUIT.compile_detector_sampler(seed=42)
    _, obs_raw = sampler.sample(shots=10, separate_observables=True)
    raw = obs_raw[:, 0].astype(int).tolist()

    print(f"  Raw observable parities   : {raw}")
    print(f"  Corrected logical outcomes: {outcomes.tolist()}")
    print(f"  Logical error rate        : {outcomes.mean():.2f}")
    print()
    print("  Raw = 1 on all shots  → X error flipped the logical Z.")
    print("  Corrected = 0         → MWPM identified and cancelled it.")

    # --- Step-by-step single shot ---
    print("\n--- Step-by-step single-shot decode ---")
    matcher = build_mwpm_decoder(SURFACE_CODE_CIRCUIT)

    sampler  = SURFACE_CODE_CIRCUIT.compile_detector_sampler(seed=0)
    det, obs = sampler.sample(shots=1, separate_observables=True)

    det_row = det[0]
    obs_val = int(obs[0, 0])
    fired   = np.where(det_row)[0].tolist()

    print(f"  Detectors fired (syndrome) : {fired}")
    print(f"  Raw observable parity      : {obs_val}")

    predicted_flip = matcher.decode(det_row)
    correction     = int(predicted_flip[0])
    result         = obs_val ^ correction

    print(f"  MWPM predicted correction  : {correction}")
    print(f"  Corrected logical Z        : {result}  "
          f"({'|1⟩  logical error present' if result else '|0⟩  no logical error'})")