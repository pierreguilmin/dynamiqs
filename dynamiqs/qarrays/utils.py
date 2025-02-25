from __future__ import annotations

import warnings
from collections.abc import Sequence

import jax.numpy as jnp
import numpy as np
from jaxtyping import ArrayLike, DTypeLike
from qutip import Qobj

from .dense_qarray import DenseQArray, array_to_qobj_list
from .layout import Layout, dense
from .qarray import QArray, QArrayLike, get_dims, isqarraylike, to_jax, to_numpy
from .sparsedia_primitives import (
    array_to_sparsedia,
    autopad_sparsedia_diags,
    shape_sparsedia,
    sparsedia_to_array,
    stack_sparsedia,
)
from .sparsedia_qarray import SparseDIAQArray

__all__ = [
    'asqarray',
    'sparsedia_from_dict',
    'stack',
    'to_jax',
    'to_numpy',
    'to_qutip',
    'isqarraylike',
]


def asqarray(
    x: QArrayLike, dims: tuple[int, ...] | None = None, layout: Layout | None = None
) -> QArray:
    """Converts a qarray-like into a qarray.

    Args:
        x: Object to convert.
        dims _(tuple of ints or None)_: Dimensions of each subsystem in the composite
            system Hilbert space tensor product. Defaults to `None` (`x.dims` if
            available, single Hilbert space `dims=(n,)` otherwise).
        layout _(dq.dense, dq.dia or None)_: Matrix layout. If `None`, the default
            layout is `dq.dense`, except for qarrays that are directly returned.

    Returns:
        Qarray representation of the input.

    See also:
        - [`dq.isqarraylike()`][dynamiqs.isqarraylike]: returns True if the input is
            a qarray-like.

    Examples:
        >>> dq.asqarray([[1, 0], [0, -1]])
        QArray: shape=(2, 2), dims=(2,), dtype=int32, layout=dense
        [[ 1  0]
         [ 0 -1]]
        >>> dq.asqarray([[1, 0], [0, -1]], layout=dq.dia)
        QArray: shape=(2, 2), dims=(2,), dtype=int32, layout=dia, ndiags=1
        [[ 1  ⋅]
         [ ⋅ -1]]
        >>> dq.asqarray([qt.sigmax(), qt.sigmay(), qt.sigmaz()])
        QArray: shape=(3, 2, 2), dims=(2,), dtype=complex64, layout=dense
        [[[ 0.+0.j  1.+0.j]
          [ 1.+0.j  0.+0.j]]
        <BLANKLINE>
         [[ 0.+0.j  0.-1.j]
          [ 0.+1.j  0.+0.j]]
        <BLANKLINE>
         [[ 1.+0.j  0.+0.j]
          [ 0.+0.j -1.+0.j]]]
    """
    if layout is None and isinstance(x, QArray):
        return x

    layout = dense if layout is None else layout
    if layout is dense:
        return _asdense(x, dims)
    else:
        return _assparsedia(x, dims)


def _asdense(x: QArrayLike, dims: tuple[int, ...] | None) -> DenseQArray:
    if isinstance(x, DenseQArray):
        return x
    elif isinstance(x, SparseDIAQArray):
        data = sparsedia_to_array(x.offsets, x.diags)
        return DenseQArray(x.dims, False, data)

    xdims = get_dims(x)
    x = to_jax(x)
    dims = init_dims(xdims, dims, x.shape)
    return DenseQArray(dims, False, x)


def _assparsedia(x: QArrayLike, dims: tuple[int, ...] | None) -> SparseDIAQArray:
    # TODO: improve this by directly extracting the diags and offsets in case
    # the Qobj is already in sparse DIA format (only for QuTiP 5)
    if isinstance(x, SparseDIAQArray):
        return x

    xdims = get_dims(x)
    x = to_jax(x)
    dims = init_dims(xdims, dims, x.shape)
    offsets, diags = array_to_sparsedia(x)
    return SparseDIAQArray(dims, False, offsets, diags)


def init_dims(
    xdims: tuple[int, ...] | None, dims: tuple[int, ...] | None, shape: tuple[int, ...]
) -> tuple[int, ...]:
    # xdims: native dims from the original object
    # dims: dims specified by the user
    # shape: object shape
    if dims is None:
        dims = (shape[-2] if shape[-2] != 1 else shape[-1],) if xdims is None else xdims
    elif xdims is not None and xdims != dims:
        # warn if `dims` argument is specified but unused
        warnings.warn(
            f'Argument `x` is already an object with `x.dims={xdims}`, but'
            f' different `dims={dims}` were specified as input. Ignoring the '
            f'provided `dims` and proceeding with the object `x.dims`.',
            stacklevel=2,
        )

    _assert_dims_match_shape(dims, shape)

    return dims


def stack(qarrays: Sequence[QArray], axis: int = 0) -> QArray:
    """Join a sequence of qarrays along a new axis.

    Warning:
        All elements of the sequence `qarrays` must have identical types, shapes and
        `dims` attributes. Additionally, when stacking qarrays of type
        `SparseDIAQArray`, all elements must have identical `offsets` attributes.

    Args:
        qarrays: Qarrays to stack.
        axis: Axis in the result along which the input qarrays are stacked.

    Returns:
        Stacked qarray.

    Examples:
        >>> dq.stack([dq.fock(3, 0), dq.fock(3, 1)])
        QArray: shape=(2, 3, 1), dims=(3,), dtype=complex64, layout=dense
        [[[1.+0.j]
          [0.+0.j]
          [0.+0.j]]
        <BLANKLINE>
         [[0.+0.j]
          [1.+0.j]
          [0.+0.j]]]
    """
    # check validity of input
    if len(qarrays) == 0:
        raise ValueError('Argument `qarrays` must contain at least one element.')
    if not all(isinstance(q, QArray) for q in qarrays):
        raise ValueError(
            'Argument `qarrays` must contain only elements of type `QArray`.'
        )
    dims = qarrays[0].dims
    if not all(q.dims == dims for q in qarrays):
        raise ValueError(
            'Argument `qarrays` must contain elements with identical `dims` attribute.'
        )
    if not all(qarray.shape == qarrays[0].shape for qarray in qarrays):
        raise ValueError('All input qarrays must have the same shape.')

    # stack inputs depending on type
    if all(isinstance(q, DenseQArray) for q in qarrays):
        data = jnp.stack([q.data for q in qarrays], axis=axis)
        return DenseQArray(dims, False, data)
    elif all(isinstance(q, SparseDIAQArray) for q in qarrays):
        offsets, diags = stack_sparsedia(
            [qarrays.offsets for qarrays in qarrays],
            [qarrays.diags for qarrays in qarrays],
            axis=axis,
        )
        return SparseDIAQArray(dims, False, offsets, diags)
    else:
        raise NotImplementedError(
            'Stacking qarrays with different types is not implemented.'
        )


def to_qutip(x: QArrayLike, dims: tuple[int, ...] | None = None) -> Qobj | list[Qobj]:
    r"""Convert a qarray-like into a QuTiP Qobj or list of Qobjs.

    Args:
        x _(qarray-like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra,
            density matrix or operator.
        dims _(tuple of ints or None)_: Dimensions of each subsystem in the composite
            system Hilbert space tensor product. Defaults to `None` (`x.dims` if
            available, single Hilbert space `dims=(n,)` otherwise).

    Returns:
        QuTiP Qobj or list of QuTiP Qobj.

    Examples:
        >>> dq.fock(3, 1)
        QArray: shape=(3, 1), dims=(3,), dtype=complex64, layout=dense
        [[0.+0.j]
         [1.+0.j]
         [0.+0.j]]
        >>> dq.to_qutip(dq.fock(3, 1))
        Quantum object: dims=[[3], [1]], shape=(3, 1), type='ket', dtype=Dense
        Qobj data =
        [[0.]
         [1.]
         [0.]]

        For a batched qarray:
        >>> rhos = dq.stack([dq.coherent_dm(16, i) for i in range(5)])
        >>> rhos.shape
        (5, 16, 16)
        >>> len(dq.to_qutip(rhos))
        5

        Note that the tensor product structure is inferred automatically for qarrays. It
        can be specified with the `dims` argument for other types.
        >>> dq.to_qutip(dq.eye(3, 2))
        Quantum object: dims=[[3, 2], [3, 2]], shape=(6, 6), type='oper', dtype=Dense, isherm=True
        Qobj data =
        [[1. 0. 0. 0. 0. 0.]
         [0. 1. 0. 0. 0. 0.]
         [0. 0. 1. 0. 0. 0.]
         [0. 0. 0. 1. 0. 0.]
         [0. 0. 0. 0. 1. 0.]
         [0. 0. 0. 0. 0. 1.]]
    """  # noqa: E501
    from .._checks import check_shape

    if isinstance(x, Qobj):
        return x
    elif isinstance(x, DenseQArray):
        return x.to_qutip()
    elif isinstance(x, SparseDIAQArray):
        return x.asdense().to_qutip()

    xdims = get_dims(x)
    x = to_jax(x)
    dims = init_dims(xdims, dims, x.shape)
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)', '(..., n, n)')
    return array_to_qobj_list(x, dims)


def sparsedia_from_dict(
    offsets_diags: dict[int, ArrayLike],
    dims: tuple[int, ...] | None = None,
    dtype: DTypeLike | None = None,
) -> SparseDIAQArray:
    """Initialize a `SparseDIAQArray` from a dictionary of offsets and diagonals.

    Args:
        offsets_diags: Dictionary where keys are offsets and values are diagonals of
            shapes _(..., n-|offset|)_ with a common batch shape between all diagonals.
        dims _(tuple of ints or None)_: Dimensions of each subsystem in the composite
            system Hilbert space tensor product. Defaults to `None` (single Hilbert
            space `dims=(n,)`).
        dtype: Data type of the qarray. If `None`, the data type is inferred from the
            diagonals.

    Returns:
        A `SparseDIAQArray` with non-zero diagonals at the specified offsets.

    Examples:
        >>> dq.sparsedia_from_dict({0: [1, 2, 3], 1: [4, 5], -1: [6, 7]})
        QArray: shape=(3, 3), dims=(3,), dtype=int32, layout=dia, ndiags=3
        [[1 4 ⋅]
         [6 2 5]
         [⋅ 7 3]]
        >>> dq.sparsedia_from_dict({0: jnp.ones((3, 2))})
        QArray: shape=(3, 2, 2), dims=(2,), dtype=float32, layout=dia, ndiags=1
        [[[1. ⋅ ]
          [ ⋅ 1.]]
        <BLANKLINE>
         [[1. ⋅ ]
          [ ⋅ 1.]]
        <BLANKLINE>
         [[1. ⋅ ]
          [ ⋅ 1.]]]
    """
    offsets = tuple(offsets_diags.keys())
    diags = [jnp.asarray(diag, dtype=dtype) for diag in offsets_diags.values()]
    diags = autopad_sparsedia_diags(offsets, diags)
    shape = shape_sparsedia(diags)
    dims = (shape[-1],) if dims is None else dims
    _assert_dims_match_shape(dims, shape)

    return SparseDIAQArray(dims, False, offsets, diags)


def _assert_dims_match_shape(dims: tuple[int, ...], shape: tuple[int, ...]):
    # check that `dims` and `shape` are compatible
    if np.prod(dims) != np.max(shape[-2:]):
        raise ValueError(
            f'Argument `dims={dims}` is incompatible with the input shape'
            f' `shape={shape}`.'
        )
