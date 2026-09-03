"""IEEE two-column figure styling.

These figures are static print output, so the interactive and dark-mode layers of
the usual visualisation guidance do not apply; the colour rules do. The
categorical palette below is validated all-pairs (not merely adjacent) for
colour-vision deficiency, because several panels overlay four curves at once:

    slots  #2a78d6 #eb6834 #1baf7a #4a3aa7
    worst all-pairs CVD dE 9.2 (deutan), normal-vision dE 16.3, on a white surface

Aqua sits at 2.82:1 against white, below the 3:1 bar, so it always ships with
relief -- a legend entry plus its own dash pattern. Every series in fact carries a
dash pattern as well as a hue, which is what keeps the figures readable in the
grayscale printing an IEEE proceedings still gets.

Analytical bounds are chrome, not data: they are drawn in muted ink so a reader
never mistakes a theoretical envelope for another measured series.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# --- geometry -------------------------------------------------------------
COL_W = 3.5          # IEEE single column, inches
COL2_W = 7.16        # IEEE full width, inches

# --- categorical series (identity) ---------------------------------------
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"]
DASHES = [(None, None), (4.0, 1.6), (1.2, 1.2), (5.0, 1.4, 1.2, 1.4)]
MARKERS = ["o", "s", "^", "D"]

# --- sequential ramp (ordered magnitude: n, sample rate, wbar) ------------
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

# --- ink ------------------------------------------------------------------
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

# --- status (used only for pass/fail annotation) --------------------------
GOOD = "#0ca30c"
CRITICAL = "#d03b3b"

FIG_DIR = Path(__file__).resolve().parent.parent / "figures"


def use_paper_style() -> None:
    """Apply the rcParams every figure script shares."""
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,          # raster output; 300 dpi is print-grade at these sizes
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "pdf.fonttype": 42,             # embed TrueType, not Type 3 (IEEE requirement)
        "ps.fonttype": 42,
        "font.family": "sans-serif",
        "font.sans-serif": ["DejaVu Sans"],
        "font.size": 8,
        "axes.titlesize": 8,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6.5,
        "mathtext.fontset": "dejavusans",
        "axes.linewidth": 0.6,
        "axes.edgecolor": AXIS,
        "axes.labelcolor": INK,
        "axes.titlecolor": INK,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.4,
        "grid.alpha": 1.0,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_2,
        "ytick.labelcolor": INK_2,
        "xtick.major.width": 0.5,
        "ytick.major.width": 0.5,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "lines.linewidth": 1.2,
        "lines.markersize": 3.0,
        "legend.frameon": True,
        "legend.framealpha": 0.92,
        "legend.edgecolor": AXIS,
        "legend.fancybox": False,
        "legend.borderpad": 0.35,
        "legend.handlelength": 2.2,
        "legend.labelspacing": 0.25,
        "legend.handletextpad": 0.5,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 0.03,
        "figure.constrained_layout.w_pad": 0.03,
    })


def series_style(i: int, marker: bool = False) -> dict:
    """Hue plus a dash pattern, so identity survives grayscale printing."""
    k = i % len(SERIES)
    st = {"color": SERIES[k], "dashes": DASHES[k]}
    if marker:
        st["marker"] = MARKERS[k]
    return st


def sequential(i: int, n: int) -> str:
    """Step ``i`` of ``n`` on the blue ramp, for ordered (not categorical) sweeps.

    Starts at index 1 so the lightest step, which recedes into the page, is only
    used where a value really is near zero.
    """
    if n <= 1:
        return SEQUENTIAL[3]
    lo, hi = 1, len(SEQUENTIAL) - 1
    return SEQUENTIAL[lo + round(i * (hi - lo) / (n - 1))]


def bound_style(**kw) -> dict:
    """Muted styling for an analytical bound -- chrome, never a data series."""
    st = {"color": INK_2, "linestyle": (0, (5, 2)), "linewidth": 0.9, "zorder": 1.5}
    st.update(kw)
    return st


def marker_line_style(**kw) -> dict:
    """Styling for a vertical event marker such as tau_c or T_dock."""
    st = {"color": MUTED, "linestyle": (0, (1.5, 1.5)), "linewidth": 0.8, "zorder": 1.2}
    st.update(kw)
    return st


def annotate_pass(ax, text: str, ok: bool = True, loc: str = "lower left") -> None:
    """Stamp a panel with the numerical verdict of the check it illustrates."""
    xy = {"lower left": (0.02, 0.03), "lower right": (0.98, 0.03),
          "upper left": (0.02, 0.97), "upper right": (0.98, 0.97)}[loc]
    ha = "right" if "right" in loc else "left"
    va = "top" if "upper" in loc else "bottom"
    ax.text(*xy, text, transform=ax.transAxes, ha=ha, va=va, fontsize=6,
            color=GOOD if ok else CRITICAL)


def save(fig, name: str) -> Path:
    """Write ``name`` as a raster PNG at ``savefig.dpi``.

    Legends drawn inside an axes are taken out of the layout calculation first.
    Constrained layout otherwise reserves room for them alongside the axes, and on
    a dense multi-panel figure that reservation can exceed the panel width, at
    which point it gives up on the whole figure and warns.
    """
    for ax in fig.axes:
        leg = ax.get_legend()
        if leg is not None:
            leg.set_in_layout(False)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    png = FIG_DIR / f"{name}.png"
    fig.savefig(png)
    plt.close(fig)
    print(f"    wrote {png.relative_to(FIG_DIR.parent)}")
    return png
