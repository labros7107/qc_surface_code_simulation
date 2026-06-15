# Surface Code Simulations

Quantum error correction simulations using [Stim](https://github.com/quantumlib/Stim) (fast stabilizer circuit simulator) and [PyMatching](https://github.com/oscarhiggott/PyMatching) (minimum-weight perfect matching decoder).

## What it does

- Simulates surface code circuits (unrotated, rotated, lattice surgery)
- Samples detection events and logical observable flips via Stim
- Decodes syndromes using MWPM via PyMatching
- Visualizes code lattices (data qubits, ancillas, stabilizers, observables)

## Quick start

```bash
# Decode a surface code circuit via MWPM
.venv/bin/python main.py circuits/unrotated_z_6x6.stim --shots 1000 --seed 42
```

### CLI options

| Argument     | Description                                     | Default   |
|-------------|-------------------------------------------------|-----------|
| `circuit`   | Path to a `.stim` circuit file                  | required  |
| `--shots`   | Number of shots to sample and decode            | `1000`    |
| `--seed`    | RNG seed for reproducible sampling              | `1`       |

### Example output

```
============================================================
  Surface code logical measurement via MWPM
============================================================
  Circuit : circuits/unrotated_z_6x6.stim
  Shots   : 1000
  Seed    : 42

  Qubits        : 121
  Detectors     : 2082
  Observables   : 1
  Measurements  : 2212
  Ticks         : 268

--- 1000-shot batch decode ---
  Raw observable parities   : [0, 0, 0, ...]
  Corrected logical outcomes: [0, 0, 0, ...]
  Logical error rate        : 0.00
```


## Dependencies

- Python ≥ 3.11
- [stim](https://github.com/quantumlib/Stim) ≥ 1.15.0
- [pymatching](https://github.com/oscarhiggott/PyMatching) ≥ 2.3.1
- [sinter](https://github.com/quantumlib/Stim) ≥ 1.15.0
- matplotlib ≥ 3.10

Install with `uv sync`.
