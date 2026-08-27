import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from hillShape import para_profile

def relative_l2(ref, pred):
    return np.linalg.norm(ref - pred) / np.linalg.norm(ref) * 100

def plot_contour(ax, x, y, U, levels, ngridx = 1000):
    xi = np.linspace(x.min(), x.max(), ngridx)
    yi = np.linspace(y.min(), y.max(), int (ngridx * (y.max()-y.min()) /(x.max()-x.min()) ) )
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, U)
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    tmp = ax.contourf(xi, yi, zi, levels=levels, cmap="RdBu")
    plt.colorbar(tmp, ax=ax, ticks = levels[::10])
    ax.set_xlim(1.,5.)
    ax.set_ylim(0.,3.)
    ax.set_xticks([2,4])
    ax.set_yticks([0,1,2,3])
    ax.set_xlabel('x/H')
    ax.set_ylabel('y/H')
    plt.tight_layout()

def plot_streamlines(ax, x, y, u, v, levels, ngridx = 1000):
    xi = np.linspace(x.min(), x.max(), ngridx)
    yi = np.linspace(y.min(), y.max(), int (ngridx * (y.max()-y.min()) /(x.max()-x.min()) ) )
    triang = tri.Triangulation(x, y)
    interpolator = tri.LinearTriInterpolator(triang, np.zeros(u.shape))
    Xi, Yi = np.meshgrid(xi, yi)
    zi = interpolator(Xi, Yi)
    tmp = ax.contourf(xi, yi, zi, levels=levels, cmap="Greys")
    Ui = griddata((x, y), u, (Xi, Yi) )
    Vi = griddata((x, y), v, (Xi, Yi) )
    strm = ax.streamplot(Xi, Yi, Ui, Vi)
    ax.set_xlim(1.,5.)
    ax.set_ylim(0.,3.)
    ax.set_xticks([2,4])
    ax.set_yticks([0,1,2,3])
    ax.set_xlabel('x/H')
    ax.set_ylabel('y/H')
    plt.tight_layout()

def plot_lines(ax, x, y, U_pred, U_star, ngrid = 1000, alpha = 1.0, legend_loc = 'upper right', bbox_to_anchor=(0.9, 0.9) ):
    yy=np.arange(0, 9, 0.01)
    ya, ha = para_profile(yy, alpha)
    ax.plot(ya, ha, 'g-')
    xend = ya[-1]
    outline = np.array([[0, 0, xend, xend], [1, 3.036, 3.036, 1]])
    ax.plot(outline[0, :], outline[1, :], 'g-')

    triang = tri.Triangulation(x, y)
    interpolator_pred = tri.LinearTriInterpolator(triang, U_pred+x)
    interpolator_star = tri.LinearTriInterpolator(triang, U_star+x)
    for i in [1.01, 2.0,3.0, 4.0, 4.99]:
        y_i = np.linspace(y.min(), y.max(), ngrid)
        x_i = i*np.ones(y_i.shape)
        z_i = interpolator_star(x_i, y_i)
        z_i_pred = interpolator_pred(x_i, y_i)
        ax.plot(z_i, y_i,'k-',label = 'DNS')
        ax.plot(z_i_pred, y_i,'r--',label = 'RSTnet')
        if i ==1.01:
            ax.legend(loc = legend_loc, frameon = False, bbox_to_anchor=bbox_to_anchor )
    ax.set_ylim(0.,3.036)
    ax.set_xticks([0,2,4,6,8,10])
    ax.set_yticks([0,1,2,3])
    ax.set_ylabel('y/H')
    plt.tight_layout()


