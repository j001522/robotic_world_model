"""
styles.py — plotnine theme, color palette, and save_plot helper.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from plotnine import (
    theme_bw,
    theme,
    element_text,
    element_line,
    element_rect,
    element_blank,
    scale_color_manual,
    scale_fill_manual,
    ggplot,
)

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Colour palette — distinct, saturated, colour-blind-friendly
# ---------------------------------------------------------------------------
# Designed for maximum contrast in both lines and legend swatches.
# Uses saturated hues that remain distinguishable under the three
# common forms of colour-vision deficiency (protanopia, deuteranopia,
# tritanopia).
# Order: B0, Bp, R0, R6, Rp, Rv, E0 + extras
PALETTE_ORDERED = [
    "#0077BB",  # strong blue   – B0
    "#33BBEE",  # vivid cyan    – Bp
    "#009988",  # teal          – R0
    "#EE7733",  # orange        – R6
    "#CC3311",  # vermillion    – Rp
    "#EE3377",  # magenta       – Rv
    "#888888",  # mid-grey      – E0
    "#332288",  # indigo        – extra
    "#882255",  # wine          – extra
    "#117733",  # forest green  – extra
]

# Short label → colour
CONDITION_COLOURS: dict[str, str] = {
    "B0": PALETTE_ORDERED[0],
    "Bp": PALETTE_ORDERED[1],
    "R0": PALETTE_ORDERED[2],
    "R6": PALETTE_ORDERED[3],
    "Rp": PALETTE_ORDERED[4],
    "Rv": PALETTE_ORDERED[5],
    "E0": PALETTE_ORDERED[6],
}


def _build_colour_map(conditions: list[str]) -> dict[str, str]:
    """Return a colour map for a list of (short) condition labels."""
    cmap: dict[str, str] = {}
    extra_idx = 0
    for c in sorted(set(conditions)):
        if c in CONDITION_COLOURS:
            cmap[c] = CONDITION_COLOURS[c]
        else:
            cmap[c] = PALETTE_ORDERED[min(extra_idx + 7, len(PALETTE_ORDERED) - 1)]
            extra_idx += 1
    return cmap


def colour_scales(conditions: list[str]):
    """Return a (scale_color_manual, scale_fill_manual) pair."""
    cmap = _build_colour_map(conditions)
    return (
        scale_color_manual(values=cmap),
        scale_fill_manual(values=cmap),
    )


def build_colour_map(conditions: list[str]) -> dict[str, str]:
    """Public helper to build a colour map for conditions."""
    return _build_colour_map(conditions)


# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
def paper_theme(base_size: int = 11):
    """Return a clean theme suitable for paper figures."""
    return (
        theme_bw(base_size=base_size)
        + theme(
            plot_title=element_text(size=base_size + 1, weight="bold"),
            axis_title=element_text(size=base_size),
            axis_text=element_text(size=base_size - 1),
            legend_title=element_text(size=base_size - 1, weight="bold"),
            legend_text=element_text(size=base_size - 2),
            legend_position="bottom",
            legend_direction="horizontal",
            legend_box_spacing=0.02,
            legend_key_size=14,
            legend_key_spacing_x=4,
            legend_margin=0,
            legend_box_margin=0,
            panel_grid_minor=element_blank(),
            strip_background=element_rect(fill="#F0F0F0", colour="grey"),
            strip_text=element_text(size=base_size - 1),
            figure_size=(6, 3.8),
            plot_margin=0.02,
        )
    )


def paper_theme_no_title(base_size: int = 16):
    """Paper theme with larger text and no titles or facet labels."""
    return (
        theme_bw(base_size=base_size)
        + theme(
            plot_title=element_blank(),
            plot_subtitle=element_blank(),
            axis_title=element_text(size=base_size + 1),
            axis_text=element_text(size=base_size - 1),
            legend_title=element_blank(),
            legend_text=element_text(size=base_size - 1),
            legend_position="bottom",
            legend_direction="horizontal",
            legend_box_spacing=0.02,
            legend_key_size=16,
            legend_key_spacing_x=4,
            legend_margin=0,
            legend_box_margin=0,
            panel_grid_minor=element_blank(),
            strip_background=element_rect(fill="#F0F0F0", colour="grey"),
            strip_text=element_blank(),
            figure_size=(6, 3.8),
            plot_margin=0.02,
        )
    )


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------
def save_plot(
    p: ggplot,
    output_dir: Path,
    name: str,
    width: float = 6,
    height: float = 3.8,
    dpi: int = 300,
    formats: Optional[list[str]] = None,
    caption: Optional[str] = None,
) -> list[Path]:
    """Save a plotnine ggplot to *output_dir/name.{fmt}*.

    Parameters
    ----------
    formats : list[str]
        File formats to save. Default: ``["pdf"]``.
    caption : str, optional
        Caption text to save as {name}_caption.txt alongside the plot.

    Returns list of written paths (including caption file if provided).
    """
    if formats is None:
        formats = ["pdf"]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        try:
            p.save(
                path,
                width=width,
                height=height,
                dpi=dpi,
                verbose=False,
                bbox_inches="tight",
            )
            log.info("Saved %s", path)
            paths.append(path)
        except Exception as exc:
            log.error("Failed to save %s: %s", path, exc)
    
    # Save caption as separate text file if provided
    if caption:
        caption_path = output_dir / f"{name}_caption.txt"
        caption_path.write_text(caption, encoding="utf-8")
        log.info("Saved caption %s", caption_path)
        paths.append(caption_path)
    
    return paths


def save_plot_grid(
    plots: list[ggplot],
    output_dir: Path,
    name: str,
    ncols: int = 1,
    width: float = 12,
    height: float = 3.8,
    dpi: int = 300,
    formats: Optional[list[str]] = None,
    caption: Optional[str] = None,
) -> list[Path]:
    """Save a grid of plotnine ggplots into a single figure.

    Parameters
    ----------
    plots : list[ggplot]
        Plots to arrange in a grid.
    ncols : int
        Number of columns in the grid.
    width, height : float
        Output size in inches.
    """
    if formats is None:
        formats = ["pdf"]
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    if not plots:
        return paths

    # Render each plot and place its artists into a target grid
    import matplotlib.pyplot as plt

    n = len(plots)
    ncols = max(1, ncols)
    nrows = (n + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(width, height))
    if nrows == 1 and ncols == 1:
        axes_list = [axes]
    elif nrows == 1:
        axes_list = list(axes)
    elif ncols == 1:
        axes_list = list(axes)
    else:
        axes_list = [ax for row in axes for ax in row]

    for ax in axes_list[n:]:
        ax.set_visible(False)

    for p, ax in zip(plots, axes_list[:n], strict=False):
        src_fig = p.draw()
        if not src_fig.axes:
            continue
        src_ax = src_fig.axes[0]

        # Copy artists
        for artist in list(src_ax.get_children()):
            try:
                artist.remove()
                ax.add_artist(artist)
            except Exception:
                continue

        # Copy axes settings
        ax.set_xlim(src_ax.get_xlim())
        ax.set_ylim(src_ax.get_ylim())
        ax.set_xscale(src_ax.get_xscale())
        ax.set_yscale(src_ax.get_yscale())
        ax.set_xlabel(src_ax.get_xlabel())
        ax.set_ylabel(src_ax.get_ylabel())
        ax.set_title(src_ax.get_title())
        ax.grid(False)

        # Remove the source figure to avoid memory build-up
        plt.close(src_fig)

    fig.tight_layout()

    for fmt in formats:
        path = output_dir / f"{name}.{fmt}"
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        log.info("Saved %s", path)
        paths.append(path)

    if caption:
        caption_path = output_dir / f"{name}_caption.txt"
        caption_path.write_text(caption, encoding="utf-8")
        log.info("Saved caption %s", caption_path)
        paths.append(caption_path)

    plt.close(fig)
    return paths
