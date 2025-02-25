from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from IPython.display import Image
from jaxtyping import ArrayLike
from matplotlib.axes import Axes
from matplotlib.colors import Normalize

from .._checks import check_shape
from ..qarrays.qarray import QArrayLike
from ..qarrays.utils import asqarray, to_jax
from ..utils import wigner as compute_wigner
from .utils import add_colorbar, colors, gif_indices, gifit, grid, optional_ax

__all__ = ['wigner', 'wigner_gif', 'wigner_mosaic']


@optional_ax
def plot_wigner_data(
    wigner: ArrayLike,
    xmax: float,
    ymax: float,
    *,
    ax: Axes | None = None,
    vmax: float = 2 / jnp.pi,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    colorbar: bool = True,
    cross: bool = False,
    clear: bool = False,
):
    w = to_jax(wigner)
    check_shape(w, 'wigner', '(n, n)')

    # set plot norm
    vmin = -vmax
    norm = Normalize(vmin=vmin, vmax=vmax, clip=True)

    # clip to avoid rounding errors
    w = w.clip(vmin, vmax)

    # plot
    ax.imshow(
        w.T,
        cmap=cmap,
        norm=norm,
        origin='lower',
        aspect='equal',
        interpolation=interpolation,
        extent=[-xmax, xmax, -ymax, ymax],
    )

    # axis label
    ax.set(xlabel=r'$\mathrm{Re}(\alpha)$', ylabel=r'$\mathrm{Im}(\alpha)$')

    if colorbar and not clear:
        cax = add_colorbar(ax, cmap, norm)
        if vmax == 2 / jnp.pi:
            cax.set_yticks([vmin, 0.0, vmax], labels=[r'$-2/\pi$', r'$0$', r'$2/\pi$'])

    if cross:
        ax.axhline(0.0, color=colors['grey'], ls='-', lw=0.7, alpha=0.8)
        ax.axvline(0.0, color=colors['grey'], ls='-', lw=0.7, alpha=0.8)

    if clear:
        ax.grid(False)
        ax.axis(False)


@optional_ax
def wigner(
    state: QArrayLike,
    *,
    ax: Axes | None = None,
    xmax: float = 5.0,
    ymax: float | None = None,
    vmax: float = 2 / jnp.pi,
    npixels: int = 101,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    colorbar: bool = True,
    cross: bool = False,
    clear: bool = False,
):
    r"""Plot the Wigner function of a state.

    Warning:
        Documentation redaction in progress.

    Note:
        Choose a diverging colormap `cmap` for better results.

    Warning:
        The axis scaling is chosen so that a coherent state $\ket{\alpha}$ lies at the
        coordinates $(x,y)=(\mathrm{Re}(\alpha),\mathrm{Im}(\alpha))$, which is
        different from the default behaviour of `qutip.plot_wigner()`.

    Examples:
        >>> psi = dq.coherent(16, 2.0)
        >>> dq.plot.wigner(psi)
        >>> renderfig('plot_wigner_coh')

        ![plot_wigner_coh](../../figs_code/plot_wigner_coh.png){.fig-half}

        >>> psi = (dq.coherent(16, 2) + dq.coherent(16, -2)).unit()
        >>> dq.plot.wigner(psi, xmax=4.0, ymax=2.0, colorbar=False)
        >>> renderfig('plot_wigner_cat')

        ![plot_wigner_cat](../../figs_code/plot_wigner_cat.png){.fig-half}

        >>> psi = (dq.fock(2, 0) + dq.fock(2, 1)).unit()
        >>> dq.plot.wigner(psi, xmax=2.0, cross=True)
        >>> renderfig('plot_wigner_01')

        ![plot_wigner_01](../../figs_code/plot_wigner_01.png){.fig-half}

        >>> psi = dq.coherent(32, [3, 3j, -3, -3j]).sum(0).unit()
        >>> dq.plot.wigner(psi, npixels=201, clear=True)
        >>> renderfig('plot_wigner_4legged')

        ![plot_wigner_4legged](../../figs_code/plot_wigner_4legged.png){.fig-half}
    """
    state = asqarray(state)
    check_shape(state, 'state', '(n, 1)', '(n, n)')

    ymax = xmax if ymax is None else ymax
    _, _, w = compute_wigner(state, xmax, ymax, npixels)

    plot_wigner_data(
        w,
        xmax,
        ymax,
        ax=ax,
        vmax=vmax,
        cmap=cmap,
        interpolation=interpolation,
        colorbar=colorbar,
        cross=cross,
        clear=clear,
    )


def wigner_mosaic(
    states: QArrayLike,
    *,
    n: int = 8,
    nrows: int = 1,
    w: float = 3.0,
    h: float | None = None,
    xmax: float = 5.0,
    ymax: float | None = None,
    vmax: float = 2 / jnp.pi,
    npixels: int = 101,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    cross: bool = False,
):
    r"""Plot the Wigner function of multiple states in a mosaic arrangement.

    Warning:
        Documentation redaction in progress.

    See [`dq.plot.wigner()`][dynamiqs.plot.wigner] for more details.

    Examples:
        >>> psis = [dq.fock(3, i) for i in range(3)]
        >>> dq.plot.wigner_mosaic(psis)
        >>> renderfig('plot_wigner_mosaic_fock')

        ![plot_wigner_mosaic_fock](../../figs_code/plot_wigner_mosaic_fock.png){.fig}

        >>> n = 16
        >>> a = dq.destroy(n)
        >>> H = dq.zeros(n)
        >>> jump_ops = [a @ a - 4.0 * dq.eye(n)]  # cat state inflation
        >>> psi0 = dq.coherent(n, 0)
        >>> tsave = jnp.linspace(0, 1.0, 101)
        >>> result = dq.mesolve(H, jump_ops, psi0, tsave)
        >>> dq.plot.wigner_mosaic(result.states, n=6, xmax=4.0, ymax=2.0)
        >>> renderfig('plot_wigner_mosaic_cat')

        ![plot_wigner_mosaic_cat](../../figs_code/plot_wigner_mosaic_cat.png){.fig}

        >>> n = 16
        >>> a = dq.destroy(n)
        >>> H = a.dag() @ a.dag() @ a @ a  # Kerr Hamiltonian
        >>> psi0 = dq.coherent(n, 2)
        >>> tsave = jnp.linspace(0, jnp.pi, 101)
        >>> result = dq.sesolve(H, psi0, tsave)
        >>> dq.plot.wigner_mosaic(result.states, n=25, nrows=5, xmax=4.0)
        >>> renderfig('plot_wigner_mosaic_kerr')

        ![plot_wigner_mosaic_kerr](../../figs_code/plot_wigner_mosaic_kerr.png){.fig}
    """
    states = asqarray(states)
    check_shape(states, 'states', '(N, n, 1)', '(N, n, n)')

    nstates = len(states)
    n = min(nstates, n)

    # create grid of plot
    _, axs = grid(
        n,
        nrows=nrows,
        w=w,
        h=h,
        gridspec_kw=dict(wspace=0, hspace=0),
        sharex=True,
        sharey=True,
    )

    ymax = xmax if ymax is None else ymax
    selected_indexes = np.linspace(0, nstates, n, dtype=int)
    _, _, wig = compute_wigner(states[selected_indexes], xmax, ymax, npixels)

    # plot individual wigner
    for i, ax in enumerate(axs):
        plot_wigner_data(
            wig[i],
            ax=ax,
            xmax=xmax,
            ymax=ymax,
            vmax=vmax,
            cmap=cmap,
            interpolation=interpolation,
            colorbar=False,
            cross=cross,
            clear=False,
        )
        ax.set(xlabel='', ylabel='', xticks=[], yticks=[])


def wigner_gif(
    states: QArrayLike,
    *,
    gif_duration: float = 5.0,
    fps: int = 10,
    w: float = 5.0,
    xmax: float = 5.0,
    ymax: float | None = None,
    vmax: float = 2 / jnp.pi,
    npixels: int = 101,
    cmap: str = 'dq',
    interpolation: str = 'bilinear',
    cross: bool = False,
    clear: bool = False,
) -> Image:
    r"""Plot a GIF of the Wigner function of multiple states.

    Warning:
        Documentation redaction in progress.

    See [`dq.plot.wigner()`][dynamiqs.plot.wigner] and
    [`dq.plot.gifit()`][dynamiqs.plot.gifit] for more details.

    Examples:
        >>> n = 16
        >>> a = dq.destroy(n)
        >>> H = dq.zeros(n)
        >>> jump_ops = [a @ a - 4.0 * dq.eye(n)]  # cat state inflation
        >>> psi0 = dq.coherent(n, 0)
        >>> tsave = jnp.linspace(0, 1.0, 1001)
        >>> result = dq.mesolve(H, jump_ops, psi0, tsave)
        >>> gif = dq.plot.wigner_gif(result.states, fps=25, xmax=4.0, ymax=2.0)
        >>> rendergif(gif, 'wigner-cat')

        ![plot_wigner_gif_cat](../../figs_code/wigner-cat.gif){.fig}

        >>> n = 16
        >>> a = dq.destroy(n)
        >>> H = a.dag() @ a.dag() @ a @ a  # Kerr Hamiltonian
        >>> psi0 = dq.coherent(n, 2)
        >>> tsave = jnp.linspace(0, jnp.pi, 1001)
        >>> result = dq.sesolve(H, psi0, tsave)
        >>> gif = dq.plot.wigner_gif(
        ...     result.states, gif_duration=10.0, fps=25, xmax=4.0, clear=True
        ... )
        >>> rendergif(gif, 'wigner-kerr')

        ![plot_wigner_gif_kerr](../../figs_code/wigner-kerr.gif){.fig-half}
    """
    states = asqarray(states)
    check_shape(states, 'states', '(N, n, 1)', '(N, n, n)')

    ymax = xmax if ymax is None else ymax
    nframes = int(gif_duration * fps)
    indices = gif_indices(len(states), nframes)
    _, _, wig = compute_wigner(states[indices], xmax, ymax, npixels)

    return gifit(plot_wigner_data)(
        wig,
        w=w,
        h=ymax / xmax * w,
        xmax=xmax,
        ymax=ymax,
        vmax=vmax,
        cmap=cmap,
        interpolation=interpolation,
        colorbar=False,
        cross=cross,
        clear=clear,
        gif_duration=gif_duration,
        fps=fps,
    )
