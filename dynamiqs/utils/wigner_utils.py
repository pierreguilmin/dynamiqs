from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array, lax
from jaxtyping import ArrayLike

from .._checks import _warn_non_normalised, check_shape
from ..qarrays.qarray import QArrayLike
from ..qarrays.utils import to_jax
from .general import todm

__all__ = ['wigner']


def wigner(
    state: QArrayLike,
    xmax: float = 6.0,
    ymax: float = 6.0,
    npixels: int = 201,
    xvec: ArrayLike | None = None,
    yvec: ArrayLike | None = None,
    g: float = 2.0,
) -> tuple[Array, Array, Array]:
    r"""Compute the Wigner distribution of a ket or density matrix.

    The Wigner distribution is computed on a grid of coordinates $(x, y)$.

    Args:
        state _(qarray-like of shape (..., n, 1) or (..., n, n))_: Ket or density
            matrix.
        xmax: Maximum absolute value of the $x$ coordinate.
        ymax: Maximum absolute value of the $y$ coordinate.
        npixels: Number of pixels in each direction.
        xvec _(array-like of shape (nxvec,), optional)_: $x$ coordinates. If `None`,
            defaults to `xvec = jnp.linspace(-xmax, xmax, npixels)`.
        yvec _(array-like of shape (nyvec,), optional)_: $y$ coordinates. If `None`,
            defaults to `yvec = jnp.linspace(-ymax, ymax, npixels)`.
        g: Scaling factor of Wigner quadratures, such that $a = g(x + iy)/2$.

    Returns:
        A tuple `(xvec, yvec, w)` where

            - **xvec** _(array of shape (npixels,) or (nxvec,))_ -- $x$ coordinates, or
                `xvec` if specified.
            - **yvec** _(array of shape (npixels,) or (nyvec,))_ -- $y$ coordinates, or
                `yvec` if specified.
            - **w** _(array of shape (..., npixels, npixels) or (..., nyvec, nxvec))_ -- Wigner distribution.
    """  # noqa: E501
    state = to_jax(state)
    check_shape(state, 'state', '(..., n, 1)', '(..., n, n)')

    _warn_non_normalised(state, 'state')

    # === convert state to density matrix
    state = to_jax(todm(state))

    # === prepare xvec and yvec
    xvec = jnp.linspace(-xmax, xmax, npixels) if xvec is None else jnp.asarray(xvec)
    check_shape(xvec, 'xvec', '(n,)', subs={'n': 'nxvec'})
    yvec = jnp.linspace(-ymax, ymax, npixels) if yvec is None else jnp.asarray(yvec)
    check_shape(yvec, 'yvec', '(n,)', subs={'n': 'nyvec'})

    return xvec, yvec, _wigner(state, xvec, yvec, g)


@partial(jax.jit)
@partial(jnp.vectorize, signature='(n,m)->(k,l)', excluded={1, 2, 3})
def _wigner(state: Array, xvec: Array, yvec: Array, g: float = 2.0) -> Array:
    """Compute the wigner distribution of a density matrix using the iterative method
    of QuTiP based on the Clenshaw summation algorithm.
    """
    n = state.shape[-1]

    x, p = jnp.meshgrid(xvec, yvec, indexing='ij')
    a = 0.5 * g * (x + 1.0j * p)
    a2 = jnp.abs(a) ** 2

    w = 2 * state[0, -1] * jnp.ones_like(a)
    state = state * (2 * jnp.ones((n, n)) - jnp.eye(n))

    def loop(i: int, w: Array) -> Array:
        i = n - 2 - i
        w = w * (2 * a * (i + 1) ** (-0.5))
        return w + _laguerre_series(i, 4 * a2, state, n)

    w = lax.fori_loop(0, n - 1, loop, w)

    return w.real * jnp.exp(-2 * a2) * 0.5 * g**2 / jnp.pi


def _diag_element(mat: jnp.array, diag: int, element: int) -> float:
    r"""Return the element of a matrix `mat` at `jnp.diag(mat, diag)[element]`.
    This function is jittable for `diag` while it is not for the `jnp.diag` version.
    """
    assert mat.shape[0] == mat.shape[1], 'Matrix must be square.'
    n = mat.shape[0]
    element = jax.lax.select(element < 0, n - jnp.abs(diag) - jnp.abs(element), element)
    return mat[jnp.maximum(-diag, 0) + element, jnp.maximum(diag, 0) + element]


def _laguerre_series(i: int, x: Array, rho: Array, n: int) -> Array:
    r"""Evaluate a polynomial series of the form `$\sum_n c_n L_n^i$` where `$L_n$` is
    such that `$L_n^i = (-1)^n \sqrt(i!n!/(i+n)!) LaguerreL[n,i,x]$`.
    """

    def n_1() -> Array:
        return _diag_element(rho, i, 0) * jnp.ones_like(x)

    def n_2() -> Array:
        c0 = _diag_element(rho, i, 0)
        c1 = _diag_element(rho, i, 1)
        return (c0 - c1 * (i + 1 - x) * (i + 1) ** (-0.5)) * jnp.ones_like(x)

    def n_other() -> Array:
        cm2 = _diag_element(rho, i, -2)
        cm1 = _diag_element(rho, i, -1)
        y0 = cm2 * jnp.ones_like(x)
        y1 = cm1 * jnp.ones_like(x)

        def loop(j: int, args: tuple[Array, Array]) -> tuple[Array, Array]:
            k = n + 1 - i - j
            y0, y1 = args
            ckm1 = _diag_element(rho, i, -j)
            y0, y1 = (
                ckm1 - y1 * (k * (i + k) / ((i + k + 1) * (k + 1))) ** 0.5,
                y0 - y1 * (i + 2 * k - x + 1) * ((i + k + 1) * (k + 1)) ** -0.5,
            )

            return y0, y1

        y0, y1 = lax.fori_loop(3, n + 1 - i, loop, (y0, y1))

        return y0 - y1 * (i + 1 - x) * (i + 1) ** (-0.5)

    return lax.cond(n - i == 1, n_1, lambda: lax.cond(n - i == 2, n_2, n_other))

