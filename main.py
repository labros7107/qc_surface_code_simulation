"""
Surface code MWPM decoder

Usage
-----
    python main.py <circuit_file> [--shots N] [--seed S]

Examples
--------
    python main.py circuits/unrotated_z_6x6.stim
    python main.py circuits/surface_code_rotated_split.stim --shots 10000 --seed 42

    Existing cicuits in /circuits can be used as templates to introduce X or Z errors to observe MWPM working.
"""

import argparse
import sys
from pathlib import Path

# Add lib/ to the import path so we can import mwpm.py
sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))

import stim
from mwpm import exec_mwpm


def main():
    parser = argparse.ArgumentParser(
        description="Decode a stim surface-code circuit via MWPM and display results.",
    )
    parser.add_argument(
        "circuit",
        type=str,
        help="Path to a .stim circuit file (must contain DETECTOR and OBSERVABLE_INCLUDE annotations).",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=100,
        help="Number of shots to sample and decode (default: 1000).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1,
        help="RNG seed for reproducibility (default: 1).",
    )
    args = parser.parse_args()

    circuit_path = Path(args.circuit)
    if not circuit_path.exists():
        print(f"Error: file not found — {circuit_path}", file=sys.stderr)
        sys.exit(1)

    print("=" * 60)
    print("  Surface code logical measurement via MWPM")
    print("=" * 60)
    print(f"  Circuit : {circuit_path}")
    print(f"  Shots   : {args.shots}")
    print(f"  Seed    : {args.seed}")
    print()

    # Load the circuit
    circuit = stim.Circuit.from_file(str(circuit_path))

    # Print circuit summary
    print(f"  Qubits        : {circuit.num_qubits}")
    print(f"  Detectors     : {circuit.num_detectors}")
    print(f"  Observables   : {circuit.num_observables}")
    print(f"  Measurements  : {circuit.num_measurements}")
    print(f"  Ticks         : {circuit.num_ticks}")
    print()

    # Run MWPM
    exec_mwpm(circuit, args.shots, args.seed)


if __name__ == "__main__":
    main()
