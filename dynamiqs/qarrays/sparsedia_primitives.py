from collections import defaultdict
from collections.abc import Sequence
from functools import partial, reduce

import jax.numpy as jnp
import numpy as np
from jax._src.core import concrete_or_error
from jaxtyping import Array


def _sparsedia_slice(offset: int) -> slice:
    # Return the slice that selects the non-zero elements of a diagonal of given offset.
    # For example, a diagonal with offset 2 is stored as [0, 0, a, b, ..., z], and
    # _sparsedia_slice(2) will return the slice(2, None) to select [a, b, ..., z].
    return slice(offset, None) if offset >= 0 else slice(None, offset)


def transpose_sparsedia(
    offsets: tuple[int, ...], diags: Array
) -> tuple[tuple[int, ...], Array]:
    out_diags = jnp.zeros_like(diags)
    out_offsets = tuple(-x for x in offsets)

    for i, offset in enumerate(offsets):
        in_slice = _sparsedia_slice(offset)
        out_slice = _sparsedia_slice(-offset)
        out_diags = out_diags.at[..., i, out_slice].set(diags[..., i, in_slice])

    return out_offsets, out_diags


def reshape_sparsedia(
    offsets: tuple[int, ...], diags: Array, shape: tuple[int, ...]
) -> tuple[tuple[int, ...], Array]:
    shape = (*shape[:-2], len(offsets), diags.shape[-1])
    out_diags = jnp.reshape(diags, shape)
    return offsets, out_diags


def broadcast_sparsedia(
    offsets: tuple[int, ...], diags: Array, shape: tuple[int, ...]
) -> tuple[tuple[int, ...], Array]:
    shape = (*shape[:-2], len(offsets), diags.shape[-1])
    out_diags = jnp.broadcast_to(diags, shape)
    return offsets, out_diags


def powm_sparsedia(
    offsets: tuple[int, ...], diags: Array, n: int
) -> tuple[tuple[int, ...], Array]:
    if n == 0:
        out_offsets = (0,)
        out_diags = jnp.ones((*diags.shape[:-2], 1, diags.shape[-1]))
        return out_offsets, out_diags
    if n == 1:
        return offsets, diags
    else:
        n1_offsets, n1_diags = powm_sparsedia(offsets, diags, n - 1)
        return matmul_sparsedia_sparsedia(offsets, diags, n1_offsets, n1_diags)


def trace_sparsedia(offsets: tuple[int, ...], diags: Array) -> Array:
    main_diag_mask = np.asarray(offsets) == 0
    if np.any(main_diag_mask):
        return jnp.sum(diags[..., main_diag_mask, :], axis=(-1, -2))
    else:
        return jnp.zeros(diags.shape[:-2])


def mul_sparsedia_sparsedia(
    left_offsets: tuple[int, ...],
    left_diags: Array,
    right_offsets: tuple[int, ...],
    right_diags: Array,
) -> tuple[tuple[int, ...], Array]:
    # we check that the offsets are unique, as they should with the class __init__
    assert len(set(left_offsets)) == len(left_offsets)
    assert len(set(right_offsets)) == len(right_offsets)

    # compute the output offsets as the intersection of offsets
    out_offsets, left_ind, right_ind = np.intersect1d(
        left_offsets, right_offsets, assume_unique=True, return_indices=True
    )

    # initialize the output diagonals
    batch_shape = jnp.broadcast_shapes(left_diags.shape[:-2], right_diags.shape[:-2])
    out_shape = (*batch_shape, len(out_offsets), left_diags.shape[-1])
    dtype = jnp.promote_types(left_diags.dtype, right_diags.dtype)
    out_diags = jnp.zeros(out_shape, dtype=dtype)

    # loop over each output offset and fill the output
    for i in range(len(out_offsets)):
        out_diag = left_diags[..., left_ind[i], :] * right_diags[..., right_ind[i], :]
        out_diags = out_diags.at[..., i, :].set(out_diag)

    return _numpy_to_tuple(out_offsets), out_diags


def _numpy_to_tuple(x: np.ndarray) -> tuple:
    assert x.ndim == 1
    return tuple([sub_x.item() for sub_x in x])


def mul_sparsedia_array(
    offsets: tuple[int, ...], diags: Array, array: Array
) -> tuple[tuple[int, ...], Array]:
    # initialize the output diagonals
    batch_shape = jnp.broadcast_shapes(diags.shape[:-2], array.shape[:-2])
    out_shape = (*batch_shape, len(offsets), diags.shape[-1])
    dtype = jnp.promote_types(diags.dtype, array.dtype)
    out_diags = jnp.zeros(out_shape, dtype=dtype)

    # loop over each diagonal of the sparse matrix and fill the output
    for i, offset in enumerate(offsets):
        in_slice = _sparsedia_slice(offset)
        other_diag = jnp.diagonal(array, offset=offset, axis1=-2, axis2=-1)
        out_diag = other_diag * diags[..., i, in_slice]
        out_diags = out_diags.at[..., i, in_slice].set(out_diag)

    return offsets, out_diags


def sparsedia_to_array(offsets: tuple[int, ...], diags: Array) -> Array:
    out = jnp.zeros(shape_sparsedia(diags), dtype=diags.dtype)
    for i, offset in enumerate(offsets):
        out += _vectorized_diag(diags[..., i, :], offset)
    return out


def shape_sparsedia(diags: Array) -> tuple[int, ...]:
    n = diags.shape[-1]
    return (*diags.shape[:-2], n, n)


@partial(jnp.vectorize, signature='(n)->(n,n)', excluded={1})
def _vectorized_diag(diag: Array, offset: int) -> Array:
    return jnp.diag(diag[_sparsedia_slice(offset)], k=offset)


def array_to_sparsedia(x: Array) -> tuple[tuple[int, ...], Array]:
    concrete_or_error(None, x, '`array_to_sparsedia` does not support tracing.')
    offsets = _find_offsets(x)
    diags = _construct_diags(offsets, x)
    return offsets, diags


def _find_offsets(x: Array) -> tuple[int, ...]:
    indices = np.nonzero(x)
    return _numpy_to_tuple(np.unique(indices[-1] - indices[-2]))


def _construct_diags(offsets: tuple[int, ...], x: Array) -> Array:
    n = x.shape[-1]
    diags = jnp.zeros((*x.shape[:-2], len(offsets), n), dtype=x.dtype)

    for i, offset in enumerate(offsets):
        diag = jnp.diagonal(x, offset=offset, axis1=-2, axis2=-1)
        diags = diags.at[..., i, _sparsedia_slice(offset)].set(diag)

    return diags


def add_sparsedia_sparsedia(
    left_offsets: tuple[int, ...],
    left_diags: Array,
    right_offsets: tuple[int, ...],
    right_diags: Array,
) -> tuple[tuple[int, ...], Array]:
    # compute the output offsets
    out_offsets = np.union1d(left_offsets, right_offsets).astype(int)

    # initialize the output diagonals
    batch_shape = jnp.broadcast_shapes(left_diags.shape[:-2], right_diags.shape[:-2])
    diags_shape = (*batch_shape, len(out_offsets), left_diags.shape[-1])
    dtype = jnp.promote_types(left_diags.dtype, right_diags.dtype)
    out_diags = jnp.zeros(diags_shape, dtype=dtype)

    # loop over each offset and fill the output
    for i, offset in enumerate(out_offsets):
        if offset in left_offsets:
            left_diag = left_diags[..., left_offsets.index(offset), :]
            out_diags = out_diags.at[..., i, :].add(left_diag)
        if offset in right_offsets:
            right_diag = right_diags[..., right_offsets.index(offset), :]
            out_diags = out_diags.at[..., i, :].add(right_diag)

    return _numpy_to_tuple(out_offsets), out_diags


def matmul_sparsedia_sparsedia(
    left_offsets: tuple[int, ...],
    left_diags: Array,
    right_offsets: tuple[int, ...],
    right_diags: Array,
) -> tuple[tuple[int, ...], Array]:
    n = left_diags.shape[-1]
    batch_shape = jnp.broadcast_shapes(left_diags.shape[:-2], right_diags.shape[:-2])
    dtype = jnp.promote_types(left_diags.dtype, right_diags.dtype)
    diag_dict = defaultdict(lambda: jnp.zeros((*batch_shape, n), dtype=dtype))

    for i, loffset in enumerate(left_offsets):
        for j, roffset in enumerate(right_offsets):
            out_offset = loffset + roffset

            if abs(out_offset) > n - 1:
                continue

            lslice = _sparsedia_slice(-roffset)
            rslice = _sparsedia_slice(roffset)
            diag = left_diags[..., i, lslice] * right_diags[..., j, rslice]
            diag_dict[out_offset] = diag_dict[out_offset].at[..., rslice].add(diag)

    out_offsets = tuple(sorted(diag_dict.keys()))
    if len(out_offsets) == 0:
        # edge case where the result is a zero matrix
        out_diags = jnp.zeros((*batch_shape, 0, n), dtype=dtype)
    else:
        out_diags = jnp.stack([diag_dict[offset] for offset in out_offsets])
        out_diags = jnp.moveaxis(out_diags, 0, -2)
    return out_offsets, out_diags


def matmul_sparsedia_array(
    offsets: tuple[int, ...], diags: Array, array: Array
) -> Array:
    batch_shape = jnp.broadcast_shapes(diags.shape[:-2], array.shape[:-2])
    out_shape = (*batch_shape, diags.shape[-1], array.shape[-1])
    dtype = jnp.promote_types(diags.dtype, array.dtype)
    out = jnp.zeros(out_shape, dtype=dtype)
    for i, offset in enumerate(offsets):
        slice_in = _sparsedia_slice(offset)
        slice_out = _sparsedia_slice(-offset)
        tmp = diags[..., i, slice_in, None] * array[..., slice_in, :]
        out = out.at[..., slice_out, :].add(tmp)

    return out


def matmul_array_sparsedia(
    array: Array, offsets: tuple[int, ...], diags: Array
) -> Array:
    batch_shape = jnp.broadcast_shapes(array.shape[:-2], diags.shape[:-2])
    out_shape = (*batch_shape, array.shape[-2], diags.shape[-1])
    dtype = jnp.promote_types(array.dtype, diags.dtype)
    out = jnp.zeros(out_shape, dtype=dtype)
    for i, offset in enumerate(offsets):
        slice_in = _sparsedia_slice(offset)
        slice_out = _sparsedia_slice(-offset)
        tmp = array[..., :, slice_out] * diags[..., i, None, slice_in]
        out = out.at[..., :, slice_in].add(tmp)

    return out


def and_sparsedia_sparsedia(
    left_offsets: tuple[int, ...],
    left_diags: Array,
    right_offsets: tuple[int, ...],
    right_diags: Array,
) -> tuple[tuple[int, ...], Array]:
    # compute new offsets
    n = right_diags.shape[-1]
    left_offsets = np.asarray(left_offsets)
    right_offsets = np.asarray(right_offsets)
    out_offsets = _numpy_to_tuple(np.ravel(left_offsets[:, None] * n + right_offsets))

    # compute new diagonals
    out_diags = jnp.kron(left_diags, right_diags)

    # merge duplicate offsets and return
    out_offsets, out_diags = _compress_sparsedia(out_offsets, out_diags)
    return out_offsets, out_diags


def _compress_sparsedia(
    offsets: tuple[int, ...], diags: Array
) -> tuple[tuple[int, ...], Array]:
    # compute unique offsets
    out_offsets, inverse_ind = np.unique(offsets, return_inverse=True)

    # initialize output diagonals
    diags_shape = (*diags.shape[:-2], len(out_offsets), diags.shape[-1])
    out_diags = jnp.zeros(diags_shape, dtype=diags.dtype)

    # loop over each offset and fill the output
    for i in range(len(out_offsets)):
        mask = inverse_ind == i
        diag = jnp.sum(diags[..., mask, :], axis=-2)
        out_diags = out_diags.at[..., i, :].set(diag)

    return _numpy_to_tuple(out_offsets), out_diags


def stack_sparsedia(
    offsets_sequence: Sequence[tuple[int, ...]],
    diags_sequence: Sequence[Array],
    axis: int,
) -> tuple[tuple[int, ...], Array]:
    # compute unique offsets of the output
    out_offsets = np.asarray(reduce(np.union1d, offsets_sequence))
    offset_to_index = {offset: idx for idx, offset in enumerate(out_offsets)}

    # prepare output diagonals with the correct shape and dtype
    dtype = reduce(jnp.promote_types, [diags.dtype for diags in diags_sequence])
    in_shape = diags_sequence[0].shape
    out_shape = (len(diags_sequence), *in_shape[:-2], len(out_offsets), in_shape[-1])
    out_diags = jnp.zeros(out_shape, dtype=dtype)
    for i, (offsets, diags) in enumerate(
        zip(offsets_sequence, diags_sequence, strict=True)
    ):
        for j, offset in enumerate(offsets):
            idx = offset_to_index[offset]
            out_diags = out_diags.at[i, ..., idx, :].add(diags[..., j, :])

    # move the stack axis to the correct position
    out_diags = jnp.moveaxis(out_diags, 0, axis)
    return _numpy_to_tuple(out_offsets), out_diags


def autopad_sparsedia_diags(
    offsets: tuple[int, ...], diags: Sequence[Array]
) -> tuple[Array]:
    # stack diags in a square matrix by padding each according to its offset
    pads_width = [(abs(k), 0) if k >= 0 else (0, abs(k)) for k in offsets]
    diags = [
        jnp.pad(diag, pad_width)
        for pad_width, diag in zip(pads_width, diags, strict=True)
    ]
    dtype = reduce(jnp.promote_types, [diag.dtype for diag in diags])
    diags = jnp.stack(diags, dtype=dtype)
    return jnp.moveaxis(diags, 0, -2)
