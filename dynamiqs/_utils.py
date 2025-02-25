from __future__ import annotations

from typing import Any, get_args

import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, PyTree, ScalarLike


def type_str(type: Any) -> str:  # noqa: A002
    if type.__module__ in ('builtins', '__main__'):
        return f'`{type.__name__}`'
    else:
        return f'`{type.__module__}.{type.__name__}`'


def obj_type_str(x: Any) -> str:
    return type_str(type(x))


def get_default_dtype() -> jnp.float32 | jnp.float64:
    default_dtype = jnp.array(0.0).dtype
    return jnp.float64 if default_dtype == jnp.float64 else jnp.float32


def cdtype() -> jnp.complex64 | jnp.complex128:
    # the default dtype for complex arrays is determined by the default floating point
    # dtype
    dtype = get_default_dtype()
    if dtype is jnp.float32:
        return jnp.complex64
    elif dtype is jnp.float64:
        return jnp.complex128
    else:
        raise ValueError(f'Data type `{dtype.dtype}` is not yet supported.')


def tree_str_inline(x: PyTree) -> str:
    # return an inline formatting of a pytree as a string
    return eqx.tree_pformat(x, indent=0).replace('\n', '').replace(',', ', ')


def concatenate_sort(*args: Array | None) -> Array | None:
    args = [x for x in args if x is not None]
    if len(args) == 0:
        return None
    return jnp.sort(jnp.concatenate(args))


def is_batched_scalar(y: ArrayLike) -> bool:
    # check if a qarray-like is a scalar or a set of scalars of shape (..., 1, 1)
    return isinstance(y, get_args(ScalarLike)) or (
        isinstance(y, get_args(ArrayLike))
        and (
            y.ndim == 0
            or (y.ndim == 1 and y.shape == (1,))
            or (y.ndim > 1 and y.shape[-2:] == (1, 1))
        )
    )


def check_compatible_dims(dims1: tuple[int, ...], dims2: tuple[int, ...]):
    if dims1 != dims2:
        raise ValueError(
            f'Qarrays have incompatible dimensions. Got {dims1} and {dims2}.'
        )
