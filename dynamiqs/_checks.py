from __future__ import annotations

import warnings
from dynamiqs._utils import _get_default_dtype
import equinox as eqx
import jax.numpy as jnp
import dynamiqs as dq
from jax import Array

from .qarrays.qarray import QArray

_is_perfect_square = lambda n: int(n**0.5) ** 2 == n


_cases = {
    '(..., n, 1)': lambda x: x.ndim >= 2 and x.shape[-1] == 1,
    '(..., 1, n)': lambda x: x.ndim >= 2 and x.shape[-2] == 1,
    '(..., n, n)': lambda x: x.ndim >= 2 and x.shape[-2] == x.shape[-1],
    '(N, ..., n, n)': lambda x: x.ndim >= 3 and x.shape[-2] == x.shape[-1],
    '(..., m, n)': lambda x: x.ndim >= 2,
    '(..., n)': lambda x: x.ndim >= 1,
    '(0,)': lambda x: x.shape == (0,),
    '(2, 1)': lambda x: x.shape == (2, 1),
    '(2, 2)': lambda x: x.shape == (2, 2),
    '(n, 1)': lambda x: x.ndim == 2 and x.shape[-1] == 1,
    '(1, n)': lambda x: x.ndim == 2 and x.shape[-2] == 1,
    '(n, n)': lambda x: x.ndim == 2 and x.shape[-2] == x.shape[-1],
    '(n,)': lambda x: x.ndim == 1,
    '(N, n, 1)': lambda x: x.ndim == 3 and x.shape[-1] == 1,
    '(N, n, n)': lambda x: x.ndim == 3 and x.shape[-2] == x.shape[-1],
    '(?, n, 1)': lambda x: 2 <= x.ndim <= 3 and x.shape[-1] == 1,
    '(?, n, n)': lambda x: 2 <= x.ndim <= 3 and x.shape[-2] == x.shape[-1],
    '(..., n^2, 1)': lambda x: x.ndim >= 2
    and _is_perfect_square(x.shape[-2])
    and x.shape[-1] == 1,
}
def _warn_non_normalised(x: Array | QArray, argname: str):
    """Warn if the input array is not properly normalised."""
    dtype = _get_default_dtype()
    rtol = 0
    if dtype is jnp.float32:
        rtol = 1e-2
    elif dtype is jnp.float64:
        rtol = 1e-6
    norm = dq.norm(x)
    if jnp.abs(norm - 1.0) > rtol :
        warnings.warn(
            f'Warning: `{argname}` is not properly normalised.' 
            f'Norm=`{norm}`, expected to be around of 1.0 with a rtol of `{rtol}` ',UserWarning, 
            stacklevel=2
        )
    
   
def has_shape(x: Array | QArray, shape: str) -> bool:
    if shape in _cases:
        return _cases[shape](x)
    else:
        raise ValueError(f'Unknown shape specification `{shape}`.')


def check_shape(
    x: Array | QArray, argname: str, *shapes: str, subs: dict[str, str] | None = None
):
    # subs is used to replace symbols in the error message, this can be used to e.g.
    # specify a shape (?, n, n) but print an error message with (nH?, n, n), by passing
    # subs={'?': 'nH?'} to replace the '?' by 'nH?' in the shape specification

    for shape in shapes:
        if has_shape(x, shape):
            return

    if len(shapes) == 1:
        shapes_str = shapes[0]
    else:
        shapes_str = ', '.join(shapes[:-1]) + ' or ' + shapes[-1]

    if subs is not None:
        for k, v in subs.items():
            shapes_str = shapes_str.replace(k, v)

    raise ValueError(
        f'Argument `{argname}` must have shape {shapes_str}, but has shape'
        f' {argname}.shape={x.shape}.'
    )


def check_times(x: Array, argname: str, allow_empty: bool = False) -> Array:
    # check that an array of time is valid (it must be a 1D array sorted in strictly
    # ascending order)

    # this function should be used as e.g. `x = check_times(x, 'x')`, and the returned
    # value should be used, otherwise the final check will be removed as part of dead
    # code elimination, see https://docs.kidger.site/equinox/api/errors/ for more
    # details

    if x.ndim != 1:
        raise ValueError(
            f'Argument {argname} must be a 1D array, but is a {x.ndim}D array.'
        )
    if not allow_empty and len(x) == 0:
        raise ValueError(f'Argument {argname} must contain at least one element.')

    # this check is written to be JIT-compatible
    return eqx.error_if(
        x,
        x[1:] < x[:-1],
        f'Argument {argname} must be sorted in strictly ascending order.',
    )


def check_type_int(x: Array | QArray, argname: str):
    if not jnp.issubdtype(x.dtype, jnp.integer):
        raise ValueError(
            f'Argument {argname} must be of type integer, but is of type'
            f' {argname}.dtype={x.dtype}.'
        )


def check_hermitian(x: QArray, argname: str) -> QArray:
    rtol, atol = 1e-5, 1e-5  # TODO: fix hard-coded tolerances
    return eqx.error_if(
        x,
        jnp.logical_not(x.isherm(rtol=rtol, atol=atol)),
        f'Argument {argname} is not hermitian.',
    )

