import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.tri as tri


def plot_contour(ax, x, y, U, levels, ngridx=1000, xlabel=r'$Re_{\theta}$', ylabel='y',
                  axis_limits=None, xticks=None, yticks=None, tick_stride=25):
    xi = np.linspace(x.min(), x.max(), ngridx)
    yi = np.linspace(y.min(), y.max(), int(ngridx * (y.max() - y.min()) / (x.max() - x.min())))
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, U)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    tmp = ax.contourf(xi, yi, zi, levels=levels, cmap="RdBu")
    cbar_kwargs = dict(ax=ax, ticks=levels[::tick_stride])
    if axis_limits is None:
        cbar_kwargs['aspect'] = 10
    plt.colorbar(tmp, **cbar_kwargs)

    if axis_limits is not None:
        ax.set_xlim(*axis_limits['x'])
        ax.set_ylim(*axis_limits['y'])
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()


def plot_streamlines(ax, x, y, u, v, levels, ngridx=1000, axis_limits=None, xticks=None, yticks=None):
    xi = np.linspace(x.min(), x.max(), ngridx)
    yi = np.linspace(y.min(), y.max(), int(ngridx * (y.max() - y.min()) / (x.max() - x.min())))
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, np.zeros(u.shape))
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    ax.contourf(xi, yi, zi, levels=levels, cmap="Greys")
    Ui = griddata((x, y), u, (Xi, Yi))
    Vi = griddata((x, y), v, (Xi, Yi))
    ax.streamplot(Xi, Yi, Ui, Vi)
    if axis_limits is not None:
        ax.set_xlim(*axis_limits['x'])
        ax.set_ylim(*axis_limits['y'])
    if xticks is not None:
        ax.set_xticks(xticks)
    if yticks is not None:
        ax.set_yticks(yticks)
    ax.set_xlabel('x/H')
    ax.set_ylabel('y/H')
    plt.tight_layout()
