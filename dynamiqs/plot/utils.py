from __future__ import annotations

from collections.abc import Iterable, Sequence
from functools import wraps
from io import BytesIO
from math import ceil
from typing import TypeVar

import matplotlib
import matplotlib as mpl
import numpy as np
from cycler import cycler
from IPython.display import Image
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.axis import Axis
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import FixedLocator, MaxNLocator, MultipleLocator, NullLocator
from PIL import Image as PILImage
from tqdm import tqdm

__all__ = ['gifit', 'grid', 'mplstyle']
# __all__ = [
#     'linmap',
#     'figax',
#     'optional_ax',
#     'grid',
#     'mplstyle',
#     'integer_ticks',
#     'sample_cmap',
#     'minorticks_off',
#     'ket_ticks',
#     'bra_ticks',
#     'add_colorbar',
# ]


def linmap(x: float, a: float, b: float, c: float, d: float) -> float:
    """Map $x$ linearly from $[a,b]$ to $[c,d]$."""
    return (x - a) / (b - a) * (d - c) + c


def figax(w: float = 7.0, h: float | None = None, **kwargs) -> tuple[Figure, Axes]:
    """Return a figure with specified width and length."""
    if h is None:
        h = w / 1.6
    return plt.subplots(1, 1, figsize=(w, h), constrained_layout=True, **kwargs)


def optional_ax(func: callable) -> callable:
    """Decorator to build an `Axes` object to pass as an argument to a plot
    function if it wasn't passed by the user.

    Examples:
        Replace
        ```
        def myplot(ax=None):
            if ax is None:
                _, ax = plt.subplots(1, 1)

            # ...
        ```
        by
        ```
        @optax
        def myplot(ax=None):
            # ...
        ```
    """

    @wraps(func)
    def wrapper(  # noqa: ANN202
        *args, ax: Axes | None = None, w: float = 7.0, h: float | None = None, **kwargs
    ):
        if ax is None:
            _, ax = figax(w=w, h=h)
        return func(*args, ax=ax, **kwargs)

    return wrapper


def grid(
    n: int,
    nrows: int = 1,
    *,
    w: float = 3.0,
    h: float | None = None,
    sharexy: bool = False,
    **kwargs,
) -> tuple[Figure, Iterable[Axes]]:
    """Returns a figure and an iterator of subplots organised in a grid.

    Warning:
        Documentation redaction in progress.

    Note:
        This method is a shortcut to Matplotlib
        [`plt.subplots()`](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html#matplotlib.pyplot.subplots).

    Examples:
        For example, to plot six different curves:

        >>> x = jnp.linspace(0, 1, 101)
        >>> ys = [jnp.sin(f * 2 * jnp.pi * x) for f in range(6)]  # (6, 101)

        Replace the usual Matplotlib code

        >>> fig, axs = plt.subplots(
        ...     2, 3, figsize=(3 * 3.0, 2 * 3.0), sharex=True, sharey=True
        ... )
        >>> for i, y in enumerate(ys):
        ...     axs[i // 3][i % 3].plot(x, y)
        [...]
        >>> fig.tight_layout()

        by

        >>> _, axs = dq.plot.grid(6, 2, sharexy=True)  # 6 subplots, 2 rows
        >>> for y in ys:
        ...     next(axs).plot(x, y)
        [...]
        >>> renderfig('plot_grid')

        ![plot_grid](../../figs_code/plot_grid.png){.fig}
    """
    h = w if h is None else h
    ncols = ceil(n / nrows)
    figsize = (w * ncols, h * nrows)

    if sharexy:
        kwargs['sharex'] = True
        kwargs['sharey'] = True

    fig, axs = plt.subplots(
        nrows, ncols, figsize=figsize, constrained_layout=True, **kwargs
    )

    return fig, iter(axs.flatten())


colors = {
    'blue': '#0c5dA5',
    'red': '#ff6b6b',
    'turquoise': '#2ec4b6',
    'yellow': '#ffc463',
    'grey': '#9e9e9e',
    'purple': '#845b97',
    'brown': '#c0675c',
    'darkgreen': '#20817d',
    'darkgrey': '#666666',
}


def mplstyle(*, usetex: bool = False, dpi: int = 72):
    r"""Set custom Matplotlib style.

    Warning:
        Documentation redaction in progress.

    Examples:
        >>> x = jnp.linspace(0, 2 * jnp.pi, 101)
        >>> ys = [jnp.sin(x), jnp.sin(2 * x), jnp.sin(3 * x)]
        >>> default_mpl_style()

        Before (default Matplotlib style):

        >>> fig, ax = plt.subplots(1, 1)
        >>> for y in ys:
        ...     ax.plot(x, y)
        [...]
        >>> ax.set(xlabel=r'$x$', ylabel=r'$\sin(x)$')
        [...]
        >>> renderfig('mplstyle_before')

        ![mplstyle_before](../../figs_code/mplstyle_before.png){.fig}

        After (Dynamiqs Matplotlib style):

        >>> dq.plot.mplstyle(dpi=150)
        >>> fig, ax = plt.subplots(1, 1)
        >>> for y in ys:
        ...     ax.plot(x, y)
        [...]
        >>> ax.set(xlabel=r'$x$', ylabel=r'$\sin(x)$')
        [...]
        >>> renderfig('mplstyle_after')

        ![mplstyle_after](../../figs_code/mplstyle_after.png){.fig}
    """
    plt.rcParams.update(
        {
            # xtick
            'xtick.direction': 'in',
            'xtick.major.size': 4.5,
            'xtick.minor.size': 2.5,
            'xtick.major.width': 1.0,
            'xtick.labelsize': 12,
            'xtick.minor.visible': True,
            # ytick
            'ytick.direction': 'in',
            'ytick.major.size': 4.5,
            'ytick.minor.size': 2.5,
            'ytick.major.width': 1.0,
            'ytick.labelsize': 12,
            'ytick.minor.visible': True,
            # axes
            'axes.facecolor': 'white',
            'axes.grid': False,
            'axes.titlesize': 12,
            'axes.labelsize': 12,
            'axes.linewidth': 1.0,
            'axes.prop_cycle': cycler('color', colors.values()),
            # grid
            'grid.color': 'gray',
            'grid.linestyle': '--',
            'grid.alpha': 0.3,
            # legend
            'legend.frameon': False,
            'legend.fontsize': 12,
            # figure
            'figure.facecolor': 'white',
            'figure.dpi': dpi,
            'figure.figsize': (7, 7 / 1.6),
            # other
            'savefig.facecolor': 'white',
            'font.size': 12,
            'scatter.marker': 'o',
            'lines.linewidth': 2.0,
            # fonts
            'text.usetex': usetex,
            'text.latex.preamble': r'\usepackage{braket}',
            'font.family': 'sans-serif',
        }
    )


def integer_ticks(axis: Axis, n: int, all: bool = True):  # noqa: A002
    if all:
        axis.set_ticks(range(n))
        minorticks_off(axis)
    else:
        # let maptlotlib choose major ticks location but restrict to integers
        axis.set_major_locator(MaxNLocator(integer=True))
        # fix minor ticks to integer locations only
        axis.set_minor_locator(MultipleLocator(1))

    # format major ticks as integer
    axis.set_major_formatter(lambda x, _: f'{int(x)}')


def ket_ticks(axis: Axis):
    # fix ticks location
    axis.set_major_locator(FixedLocator(axis.get_ticklocs()))

    # format ticks as ket
    new_labels = [rf'$| {label.get_text()} \rangle$' for label in axis.get_ticklabels()]
    axis.set_ticklabels(new_labels)


def bra_ticks(axis: Axis):
    # fix ticks location
    axis.set_major_locator(FixedLocator(axis.get_ticklocs()))

    # format ticks as ket
    new_labels = [rf'$\langle {label.get_text()} |$' for label in axis.get_ticklabels()]
    axis.set_ticklabels(new_labels)


def sample_cmap(name: str, n: int, alpha: float = 1.0) -> np.ndarray:
    sampled_cmap = matplotlib.colormaps[name](np.linspace(0, 1, n))
    sampled_cmap[:, -1] = alpha
    return sampled_cmap


def minorticks_off(axis: Axis):
    axis.set_minor_locator(NullLocator())


def add_colorbar(
    ax: Axes, cmap: str, norm: Normalize, *, size: float = 0.05, pad: float = 0.05
) -> Axes:
    # insert a new axes on the right with the same height
    cax = ax.inset_axes([1 + size, 0, pad, 1])
    mappable = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    plt.colorbar(mappable=mappable, cax=cax)
    cax.grid(False)
    return cax


T = TypeVar('T')


def gif_indices(nitems: int, nframes: int) -> np.ndarray:
    # generate indices for GIF frames
    if nframes < nitems:
        return np.linspace(0, nitems - 1, nframes, dtype=int)
    else:
        return np.arange(nitems)


def gifit(
    plot_function: callable[[T, ...], None],
) -> callable[[Sequence[T], ...], Image]:
    """Transform a plot function into a new function that returns an animated GIF.

    This function takes a plot function that normally operates on a single input and
    returns a new function that creates a GIF from a sequence of inputs. The new
    function accepts two extra keyword arguments:

    - **gif_duration** _(float)_ -- GIF duration in seconds.
    - **fps** _(int)_ -- GIF frames per seconds.

    The new function returns an object of type `IPython.core.display.Image`,
    which automatically displays the GIF in Jupyter notebook environments (when the
    `Image` object is the last expression in a cell).

    ??? "Save GIF to a file"
        The returned GIF can be saved to a file with:

        ```
        with open('/path/to/file.gif').open('wb') as f:
            f.write(gif.data)
        ```

    Args:
        plot_function: Plot function which must take as first positional argument the
            input that will be sequenced over by the new function. It must create a
            matplotlib `Figure` object and not close it.

    Returns:
        A new function with the same signature as `plot_function` which accepts a
            sequence of inputs and returns a GIF by applying the original
            `plot_function` to each element in the sequence.

    Examples:
        >>> def plot_cos(phi):
        ...     x = np.linspace(0, 1.0, 501)
        ...     y = np.cos(2 * np.pi * x + phi)
        ...     plt.figure(constrained_layout=True)
        ...     plt.plot(x, y)
        >>> phis = np.linspace(0, 2 * np.pi, 101)
        >>> gif = dq.plot.gifit(plot_cos)(phis, fps=25)
        >>> gif
        <IPython.core.display.Image object>
        >>> rendergif(gif, 'cos')

        ![plot_cos](../../figs_code/cos.gif){.fig}

        >>> alphas = jnp.linspace(0.0, 3.0, 51)
        >>> states = dq.coherent(24, alphas)
        >>> gif = dq.plot.gifit(dq.plot.fock)(states, fps=25)
        >>> rendergif(gif, 'coherent_evolution')

        ![plot_coherent_evolution](../../figs_code/coherent_evolution.gif){.fig}
    """

    @wraps(plot_function)
    def wrapper(
        items: Sequence[T], *args, gif_duration: float = 5.0, fps: int = 10, **kwargs
    ) -> Image:
        nframes = int(gif_duration * fps)
        indices = gif_indices(len(items), nframes)

        frames = []
        for idx in tqdm(indices):
            plt.close()
            plot_function(items[idx], *args, **kwargs)  # plot frame
            canvas = plt.gcf().canvas
            canvas.draw()  # ensure the figure is drawn
            frame = np.array(canvas.buffer_rgba())  # capture the RGBA buffer
            frames.append(frame)

        plt.close()

        # create a BytesIO object to save the GIF in memory
        gif_buffer = BytesIO()
        # duration per frame in ms, rescaled to account for eventual duplicate frames
        duration = int(1000 / fps * nframes / len(indices))
        pil_frames = [PILImage.fromarray(frame).convert('RGBA') for frame in frames]
        pil_frames[0].save(
            gif_buffer,
            format='GIF',
            save_all=True,
            append_images=pil_frames[1:],
            duration=duration,
            loop=0,
            disposal=2,
        )

        return Image(data=gif_buffer.getvalue(), format='gif')

    return wrapper
