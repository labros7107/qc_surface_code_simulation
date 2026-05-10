"""
surface_code_diagram.py

Provides a function to visualize the lattice structure of an unrotated
surface code given a stim.Circuit (or parsed qubit/ancilla metadata).

Usage:
    import stim
    from surface_code_diagram import plot_surface_code

    circuit = stim.Circuit(\"\"\"...\"\"\")
    plot_surface_code(circuit)

    # Or save to file:
    plot_surface_code(circuit, save_path="lattice.png")
"""

import re
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Polygon
from matplotlib.lines import Line2D
import numpy as np

# ── helpers ──────────────────────────────────────────────────────────────────

def _parse_circuit(circuit) -> dict:
    """
    Extract qubit coordinates, ancilla assignments, errors, and the
    logical observable from a stim.Circuit.

    Returns a dict with:
        coords          : {qubit_index: (x, y)}
        data_qubits     : set of qubit indices
        x_ancillas      : set of qubit indices (X stabilizer ancillas)
        z_ancillas      : set of qubit indices (Z stabilizer ancillas)
        x_errors        : set of qubit indices with injected X errors
        z_errors        : set of qubit indices with injected Z errors
        observable      : list of qubit indices in the logical observable
        cx_pairs        : list of (control, target) pairs (for coupling edges)
    """
    circuit_str = str(circuit)

    # ── qubit coordinates ─────────────────────────────────────────────────
    coords: dict[int, tuple[float, float]] = {}
    for m in re.finditer(r"QUBIT_COORDS\(([\d.]+),\s*([\d.]+)\)\s+([\d,\s]+)", circuit_str):
        x, y = float(m.group(1)), float(m.group(2))
        for q in map(int, re.split(r"[\s,]+", m.group(3).strip())):
            coords[q] = (x, y)

    # ── classify qubits from circuit instructions ─────────────────────────
    # Ancillas are reset-measured (MR) each round.
    # Among ancillas, X-type are bracketed by H gates; Z-type are not.
    # Data qubits are everything that is never MR-measured.
    #
    # Strategy:
    #   1. Collect all qubits that appear in any MR instruction → ancillas.
    #   2. Collect all qubits that appear in any H instruction → h_targets.
    #   3. X ancillas = ancillas ∩ h_targets
    #   4. Z ancillas = ancillas − x_ancillas
    #   5. Data qubits = all coords − ancillas

    mr_measured: set[int] = set()
    for m in re.finditer(r"^MR\s+([\d\s]+)", circuit_str, re.MULTILINE):
        mr_measured.update(map(int, m.group(1).split()))

    h_targets: set[int] = set()
    for m in re.finditer(r"^H\s+([\d\s]+)", circuit_str, re.MULTILINE):
        h_targets.update(map(int, m.group(1).split()))

    x_ancillas:  set[int] = mr_measured & h_targets
    z_ancillas:  set[int] = mr_measured - x_ancillas
    data_qubits: set[int] = set(coords) - mr_measured

    # ── injected errors ───────────────────────────────────────────────────
    x_errors: set[int] = set()
    z_errors: set[int] = set()
    for m in re.finditer(r"X_ERROR\(1\)\s+([\d\s]+)", circuit_str):
        x_errors.update(map(int, m.group(1).split()))
    for m in re.finditer(r"Z_ERROR\(1\)\s+([\d\s]+)", circuit_str):
        z_errors.update(map(int, m.group(1).split()))

    # ── logical observable ────────────────────────────────────────────────
    # OBSERVABLE_INCLUDE(0) rec[-k] ... -> last M instruction, pick qubits
    observable: list[int] = []
    m_matches = list(re.finditer(r"^M\s+([\d\s]+)", circuit_str, re.MULTILINE))
    obs_matches = list(re.finditer(r"OBSERVABLE_INCLUDE\(0\)(.*)", circuit_str))
    if m_matches and obs_matches:
        last_m_qubits = list(map(int, m_matches[-1].group(1).split()))
        obs_str = obs_matches[-1].group(1)
        indices = [int(x) for x in re.findall(r"rec\[-(\d+)\]", obs_str)]
        for idx in indices:
            qi = len(last_m_qubits) - idx  # rec[-1] = last measured = index -1
            if 0 <= qi < len(last_m_qubits):
                observable.append(last_m_qubits[qi])

    # ── CX pairs (for drawing coupling edges) ────────────────────────────
    cx_pairs: list[tuple[int, int]] = []
    for m in re.finditer(r"^CX\s+([\d\s]+)", circuit_str, re.MULTILINE):
        nums = list(map(int, m.group(1).split()))
        for i in range(0, len(nums) - 1, 2):
            cx_pairs.append((nums[i], nums[i + 1]))

    return dict(
        coords=coords,
        data_qubits=data_qubits,
        x_ancillas=x_ancillas,
        z_ancillas=z_ancillas,
        x_errors=x_errors,
        z_errors=z_errors,
        observable=observable,
        cx_pairs=cx_pairs,
    )


def _stabilizer_plaquettes(
    coords, data_qubits, x_ancillas, z_ancillas
) -> tuple[list, list]:
    """
    For each ancilla find its neighbouring data qubits (within distance √2)
    and return the convex hull of those + the ancilla as a plaquette polygon.

    Returns (x_plaquettes, z_plaquettes) where each item is a list of (x,y)
    corner points.
    """
    all_data = {q: coords[q] for q in data_qubits if q in coords}

    def neighbours(anc_q, ancillas):
        ax, ay = coords[anc_q]
        nbrs = []
        for dq, (dx, dy) in all_data.items():
            dist = abs(dx - ax) + abs(dy - ay)
            if dist <= 1.5:          # Manhattan distance ≤ 1 (touching)
                nbrs.append((dx, dy))
        return nbrs

    def convex_hull_polygon(points):
        """Simple gift-wrapping for ≤4 points; return ordered corners."""
        if len(points) < 2:
            return points
        pts = np.array(points, dtype=float)
        centre = pts.mean(axis=0)
        angles = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
        return pts[np.argsort(angles)].tolist()

    x_plaquettes, z_plaquettes = [], []
    for anc in x_ancillas:
        if anc not in coords:
            continue
        nbrs = neighbours(anc, x_ancillas)
        if nbrs:
            x_plaquettes.append(convex_hull_polygon(nbrs))
    for anc in z_ancillas:
        if anc not in coords:
            continue
        nbrs = neighbours(anc, z_ancillas)
        if nbrs:
            z_plaquettes.append(convex_hull_polygon(nbrs))

    return x_plaquettes, z_plaquettes


# ── main function ─────────────────────────────────────────────────────────────

def plot_surface_code(
    circuit,
    *,
    title: str = "Unrotated Surface Code Lattice",
    figsize: tuple[float, float] = (9, 9),
    show_qubit_labels: bool = True,
    show_couplings: bool = True,
    show_plaquettes: bool = True,
    show_observable: bool = True,
    show_legend: bool = True,
    save_path: str | None = None,
    dpi: int = 150,
) -> plt.Figure:
    """
    Plot the lattice structure of an unrotated surface code.

    Parameters
    ----------
    circuit : stim.Circuit
        The surface code circuit to visualise.
    title : str
        Figure title.
    figsize : (width, height)
        Matplotlib figure size in inches.
    show_qubit_labels : bool
        Annotate each qubit with its index.
    show_couplings : bool
        Draw CX coupling edges between data and ancilla qubits.
    show_plaquettes : bool
        Shade X- and Z-stabilizer plaquettes.
    show_observable : bool
        Highlight the logical X observable on the data qubits.
    show_legend : bool
        Add a legend.
    save_path : str or None
        If given, save the figure to this path instead of (or as well as)
        displaying it.
    dpi : int
        Resolution for saved figure.

    Returns
    -------
    matplotlib.figure.Figure
    """

    info = _parse_circuit(circuit)
    coords      = info["coords"]
    data_qubits = info["data_qubits"]
    x_ancillas  = info["x_ancillas"]
    z_ancillas  = info["z_ancillas"]
    x_errors    = info["x_errors"]
    z_errors    = info["z_errors"]
    observable  = info["observable"]
    cx_pairs    = info["cx_pairs"]

    if not coords:
        raise ValueError("No QUBIT_COORDS found in the circuit string.")

    # ── figure setup ──────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=14, fontweight="bold", pad=14)
    ax.axis("off")

    xs = [c[0] for c in coords.values()]
    ys = [c[1] for c in coords.values()]
    margin = 0.7
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)

    # Invert y so (0,0) is top-left (matching stim coordinate convention)
    ax.invert_yaxis()

    # ── colours ───────────────────────────────────────────────────────────
    COL = dict(
        x_face   = "#B5D4F4",   # light blue — X plaquettes
        x_edge   = "#378ADD",
        z_face   = "#9FE1CB",   # light teal — Z plaquettes
        z_edge   = "#1D9E75",
        data     = "white",
        data_ec  = "#444441",
        x_anc    = "#E6F1FB",
        x_anc_ec = "#378ADD",
        z_anc    = "#E1F5EE",
        z_anc_ec = "#1D9E75",
        err_x    = "#F7C1C1",
        err_x_ec = "#A32D2D",
        err_z    = "#C0DD97",
        err_z_ec = "#3B6D11",
        obs      = "#A32D2D",
        coupling = "#88878080",
    )

    # ── plaquettes ────────────────────────────────────────────────────────
    if show_plaquettes:
        x_plaqs, z_plaqs = _stabilizer_plaquettes(
            coords, data_qubits, x_ancillas, z_ancillas
        )
        for verts in x_plaqs:
            poly = Polygon(
                verts, closed=True,
                facecolor=COL["x_face"], edgecolor=COL["x_edge"],
                linewidth=0.8, alpha=0.55, zorder=1,
            )
            ax.add_patch(poly)
        for verts in z_plaqs:
            poly = Polygon(
                verts, closed=True,
                facecolor=COL["z_face"], edgecolor=COL["z_edge"],
                linewidth=0.8, alpha=0.55, zorder=1,
            )
            ax.add_patch(poly)

    # ── coupling edges ────────────────────────────────────────────────────
    if show_couplings:
        drawn = set()
        for ctrl, tgt in cx_pairs:
            key = frozenset({ctrl, tgt})
            if key in drawn:
                continue
            drawn.add(key)
            if ctrl not in coords or tgt not in coords:
                continue
            x1, y1 = coords[ctrl]
            x2, y2 = coords[tgt]
            ax.plot(
                [x1, x2], [y1, y2],
                color=COL["coupling"], linewidth=1.0, zorder=2, solid_capstyle="round"
            )

    # ── logical observable bar ────────────────────────────────────────────
    if show_observable and observable:
        obs_coords = [coords[q] for q in observable if q in coords]
        if obs_coords:
            obs_xs = [c[0] for c in obs_coords]
            obs_ys = [c[1] for c in obs_coords]
            ax.plot(
                obs_xs, obs_ys,
                color=COL["obs"], linewidth=3.5, linestyle="--",
                zorder=3, alpha=0.8, solid_capstyle="round",
            )

    # ── qubit nodes ───────────────────────────────────────────────────────
    node_r = 0.18   # radius for data circles
    sq_h   = 0.17   # half-width for ancilla squares
    dia_h  = 0.20   # half-diagonal for Z diamond

    def _label(ax, q, x, y, color="black"):
        if show_qubit_labels:
            ax.text(
                x, y, str(q),
                ha="center", va="center",
                fontsize=7, color=color, fontweight="bold", zorder=6,
                fontfamily="monospace",
            )

    # data qubits
    for q in data_qubits:
        if q not in coords:
            continue
        x, y = coords[q]
        has_x_err = q in x_errors
        has_z_err = q in z_errors
        fc = COL["err_x"] if has_x_err else (COL["err_z"] if has_z_err else COL["data"])
        ec = COL["err_x_ec"] if has_x_err else (COL["err_z_ec"] if has_z_err else COL["data_ec"])
        lw = 1.8 if (has_x_err or has_z_err) else 1.0
        circle = plt.Circle(
            (x, y), node_r,
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=4
        )
        ax.add_patch(circle)
        lc = COL["err_x_ec"] if has_x_err else (COL["err_z_ec"] if has_z_err else "black")
        _label(ax, q, x, y, color=lc)

    # X ancilla qubits (squares)
    for q in x_ancillas:
        if q not in coords:
            continue
        x, y = coords[q]
        sq = mpatches.FancyBboxPatch(
            (x - sq_h, y - sq_h), 2 * sq_h, 2 * sq_h,
            boxstyle="square,pad=0",
            facecolor=COL["x_anc"], edgecolor=COL["x_anc_ec"],
            linewidth=1.0, zorder=4,
        )
        ax.add_patch(sq)
        _label(ax, q, x, y, color=COL["x_anc_ec"])

    # Z ancilla qubits (diamonds)
    for q in z_ancillas:
        if q not in coords:
            continue
        x, y = coords[q]
        diamond_verts = [(x, y - dia_h), (x + dia_h, y), (x, y + dia_h), (x - dia_h, y)]
        dia = Polygon(
            diamond_verts, closed=True,
            facecolor=COL["z_anc"], edgecolor=COL["z_anc_ec"],
            linewidth=1.0, zorder=4,
        )
        ax.add_patch(dia)
        _label(ax, q, x, y, color=COL["z_anc_ec"])

    # ── legend ────────────────────────────────────────────────────────────
    if show_legend:
        handles = [
            Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=COL["data"], markeredgecolor=COL["data_ec"],
                   markersize=11, label="Data qubit", markeredgewidth=1),
            Line2D([0], [0], marker="s", color="w",
                   markerfacecolor=COL["x_anc"], markeredgecolor=COL["x_anc_ec"],
                   markersize=11, label="X ancilla", markeredgewidth=1),
            Line2D([0], [0], marker="D", color="w",
                   markerfacecolor=COL["z_anc"], markeredgecolor=COL["z_anc_ec"],
                   markersize=11, label="Z ancilla", markeredgewidth=1),
            mpatches.Patch(facecolor=COL["x_face"], edgecolor=COL["x_edge"],
                           label="X stabilizer"),
            mpatches.Patch(facecolor=COL["z_face"], edgecolor=COL["z_edge"],
                           label="Z stabilizer"),
        ]
        if x_errors or z_errors:
            handles.append(
                Line2D([0], [0], marker="o", color="w",
                       markerfacecolor=COL["err_x"], markeredgecolor=COL["err_x_ec"],
                       markersize=11, label="X error", markeredgewidth=1.5)
            )
        if observable and show_observable:
            handles.append(
                Line2D([0], [0], color=COL["obs"], linewidth=2.5,
                       linestyle="--", label="Logical X̄ observable")
            )
        ax.legend(
            handles=handles, loc="upper right",
            fontsize=9, framealpha=0.92, edgecolor="#ccc",
            bbox_to_anchor=(1.0, 1.0),
        )

    # ── save / show ───────────────────────────────────────────────────────
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()

    return fig


# ── quick demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    try:
        import stim
    except ImportError:
        raise SystemExit("Install stim:  pip install stim")

    surface_code = stim.Circuit("""
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
X_ERROR(1) 20 22 24
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
TICK
M 0 2 4 6 8 10 12 14 16 18 20 22 24
OBSERVABLE_INCLUDE(0) rec[-11] rec[-12] rec[-13]
""")

    plot_surface_code(
        surface_code,
        title="Distance-5 Unrotated Surface Code",
        save_path="surface_code_lattice.png",
    )