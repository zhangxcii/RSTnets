"""Case-specific postprocessing for the periodic-hill cases."""


import importlib.util
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.tri as tri
import numpy as np

from rstnets.utils import plot_streamlines

_hills_spec = importlib.util.spec_from_file_location("phill_hills", Path(__file__).parent / "hills.py")
_hills = importlib.util.module_from_spec(_hills_spec)
_hills_spec.loader.exec_module(_hills)
para_profile = _hills.para_profile


def plot_lines(ax, x, y, U_pred, U_star, ngrid=1000, alpha=1.0, legend_loc='upper right',
                bbox_to_anchor=(0.9, 0.9)):
    yy = np.arange(0, 9, 0.01)
    ya, ha = para_profile(yy, alpha)
    ax.plot(ya, ha, 'g-')
    xend = ya[-1]
    outline = np.array([[0, 0, xend, xend], [1, 3.036, 3.036, 1]])
    ax.plot(outline[0, :], outline[1, :], 'g-')

    triang = tri.Triangulation(x, y)
    interpolator_pred = tri.LinearTriInterpolator(triang, U_pred + x)
    interpolator_star = tri.LinearTriInterpolator(triang, U_star + x)
    for i in [1.01, 2.0, 3.0, 4.0, 4.99]:
        y_i = np.linspace(y.min(), y.max(), ngrid)
        x_i = i * np.ones(y_i.shape)
        z_i = interpolator_star(x_i, y_i)
        z_i_pred = interpolator_pred(x_i, y_i)
        ax.plot(z_i, y_i, 'k-', label='DNS')
        ax.plot(z_i_pred, y_i, 'r--', label='RSTnet')
        if i == 1.01:
            ax.legend(loc=legend_loc, frameon=False, bbox_to_anchor=bbox_to_anchor)
    ax.set_ylim(0., 3.036)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_ylabel('y/H')
    plt.tight_layout()


def run(case_dir, case_config, data, model, pred, star, x_plot, y_plot, output_dir):
    axis_limits = case_config.plotting.axis_limits

    if "u" in pred and "v" in pred:
        fig, axs = plt.subplots(1, 2, figsize=(8, 3), dpi=300)
        levels = np.linspace(0, 1, 2)
        for ax, (label, uv) in zip(axs, [("RSTnet", (pred["u"], pred["v"])), ("Reference", (star["u"], star["v"]))]):
            plot_streamlines(ax, x_plot, y_plot, uv[0], uv[1], levels=levels, axis_limits=axis_limits)
            ax.set_title(label)
        plt.savefig(Path(output_dir) / "streamlines.png")
        plt.close(fig)

    if "u" in pred:
        hill_alpha = case_config.custom.get("hill_alpha", 1.0)
        fig, ax = plt.subplots(1, 1, figsize=(6, 4), dpi=300)
        plot_lines(ax, x_plot, y_plot, pred["u"] * 2, star["u"] * 2, alpha=hill_alpha)
        plt.savefig(Path(output_dir) / "hill_lines.png")
        plt.close(fig)
