from __future__ import annotations

from collections.abc import Sequence
from functools import wraps
from typing import Any

import jax
from jax._src.lib import xla_client
from jaxtyping import PyTree

from .._utils import obj_type_str
from ..method import Method, _DEAdaptiveStep
from ..qarrays.qarray import QArrayLike
from ..qarrays.utils import asqarray
from ..time_qarray import (
    ConstantTimeQArray,
    PWCTimeQArray,
    SummedTimeQArray,
    TimeQArray,
)


def astimeqarray(x: QArrayLike | TimeQArray) -> TimeQArray:
    if isinstance(x, TimeQArray):
        return x
    else:
        try:
            # same as dq.constant() but not checking the shape
            qarray = asqarray(x)
            return ConstantTimeQArray(qarray)
        except (TypeError, ValueError) as e:
            raise TypeError(
                'Argument must be a qarray-like or a time-qarray, but has type'
                f' {obj_type_str(x)}.'
            ) from e


def ispwc(x: TimeQArray) -> bool:
    # check if a time-qarray is constant or piecewise constant
    if isinstance(x, ConstantTimeQArray | PWCTimeQArray):
        return True
    elif isinstance(x, SummedTimeQArray):
        return all(ispwc(timeqarray) for timeqarray in x.timeqarrays)
    else:
        return False


def catch_xla_runtime_error(func: callable) -> callable:
    # Decorator to catch `XlaRuntimeError`` exceptions, and set a more friendly
    # exception message. Note that this will not work for jitted function, as the
    # exception code will be traced out.

    @wraps(func)
    def wrapper(*args, **kwargs):  # noqa: ANN202
        try:
            return func(*args, **kwargs)
        except xla_client.XlaRuntimeError as e:
            # === `max_steps` reached error
            eqx_max_steps_error_msg = (
                'EqxRuntimeError: The maximum number of method steps was reached. '
            )
            if eqx_max_steps_error_msg in str(e):
                default_max_steps = _DEAdaptiveStep.max_steps
                raise RuntimeError(
                    'The maximum number of method steps has been reached (the default'
                    f' value is `max_steps={default_max_steps:_}`). Try increasing'
                    ' `max_steps` with the `method` argument, e.g.'
                    ' `method=dq.method.Tsit5(max_steps=1_000_000)`.'
                ) from e
            # === other errors
            raise RuntimeError(
                'An internal JAX error interrupted the execution, please report this to'
                ' the Dynamiqs developers by opening an issue on GitHub or sending a'
                ' message on Dynamiqs Slack (links available at'
                ' https://www.dynamiqs.org/stable/community/lets-talk.html).'
            ) from e

    return wrapper


def assert_method_supported(method: Method, supported_methods: Sequence[Method]):
    if not isinstance(method, tuple(supported_methods)):
        supported_str = ', '.join(f'`{x.__name__}`' for x in supported_methods)
        raise TypeError(
            f'Method of type `{type(method).__name__}` is not supported (supported'
            f' method types: {supported_str}).'
        )


def multi_vmap(
    f: callable, in_axes: int | None | Sequence[Any], out_axes: Any, nvmap: int
) -> callable:
    """Vectorize a function multiple time over multiple shared axes (similar to
    jnp.vectorize).

    The function `f` is mapped multiple time on the input specified by `in_axes`
    and the output specified by `out_axes`. All inputs corresponding to a place where
    `in_axes` is not `None` must be broadcasted to the same shape before calling the
    returned function.

    Args:
        in_axes: Same as `in_axes` of `jax.vmap`.
        out_axes: Same as `out_axes` of `jax.vmap`.
        nvmap: Number of vectorization.

    Examples:
        >>> import jax.numpy as jnp
        >>> from dynamiqs.integrators._utils import multi_vmap
        >>>
        >>> def func(x, y):
        ...     return x.T @ y.T
        >>>
        >>> n = 2
        >>>
        >>> # vmap twice over x
        >>> x = jnp.ones((3, 4, 2, 2))
        >>> y = jnp.ones((2, 2))
        >>> f = multi_vmap(func, (0, None), 0, 2)
        >>> f(x, y).shape
        (3, 4, 2, 2)
        >>>
        >>> # vmap twice over x and y
        >>> y = jnp.ones((4, 2, 2))
        >>> f = multi_vmap(func, (0, 0), 0, 2)
        >>> x, y = jnp.broadcast_arrays(x, y)
        >>> f(x, y).shape
        (3, 4, 2, 2)
        >>>
        >>> # vmap three times over x and y
        >>> y = jnp.ones((5, 3, 1, 2, 2))
        >>> f = multi_vmap(func, (0, 0), 0, 3)
        >>> x, y = jnp.broadcast_arrays(x, y)
        >>> f(x, y).shape
        (5, 3, 4, 2, 2)
    """
    for _ in range(nvmap):
        f = jax.vmap(f, in_axes=in_axes, out_axes=out_axes)
    return f


def cartesian_vmap(
    f: callable, in_axes: int | None | Sequence[Any], out_axes: Any, nvmap: PyTree[int]
) -> callable:
    """Vectorize a function multiple time over distinct axes.

    The function `f` is mapped multiple time over on each input specified by `nvmap`
    and the output specified by `out_axes`. All inputs corresponding to a place where
    `in_axes` is not `None` must be broadcasted to the same shape before calling the
    returned function.

    Args:
        in_axes: Same as `in_axes` of `jax.vmap`.
        out_axes: Same as `out_axes` of `jax.vmap`.
        nvmap: Number of vectorization for each subtree.

    Examples:
        >>> import jax.numpy as jnp
        >>> import equinox as eqx
        >>> from dynamiqs.integrators._utils import cartesian_vmap
        >>>
        >>> def func(x, y):
        ...     return x.T @ y.T
        >>>
        >>> # vmap over all combinations of x and y
        >>> x = jnp.ones((3, 4, 5, 2, 2))
        >>> y = jnp.ones((6, 7, 2, 2))
        >>> f = cartesian_vmap(func, (0, 0), 0, (3, 2))
        >>> f(x, y).shape
        (3, 4, 5, 6, 7, 2, 2)
    """
    keyleaf = jax.tree_util.tree_leaves_with_path(nvmap)

    # apply successive vmaps in reverse order
    for path, n in keyleaf[::-1]:
        if n > 0:
            # set all elements `in_axes` to `None` except for a specific subpart
            keep_path_only = lambda cpath, x, path=path: (
                x if cpath[: len(path)] == path else None
            )
            in_axes_single = jax.tree_util.tree_map_with_path(keep_path_only, in_axes)
            for _ in range(n):
                f = jax.vmap(f, in_axes=in_axes_single, out_axes=out_axes)

    return f
