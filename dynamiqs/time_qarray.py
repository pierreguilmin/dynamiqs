from __future__ import annotations

import functools as ft
from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax import Array, lax
from jaxtyping import ArrayLike, PyTree, Scalar, ScalarLike

from ._checks import check_shape, check_times
from ._utils import _concatenate_sort, cdtype, obj_type_str
from .qarrays.layout import Layout, dia, promote_layouts
from .qarrays.qarray import QArray, QArrayLike, isqarraylike
from .qarrays.utils import asqarray

__all__ = ['TimeQArray', 'constant', 'modulated', 'pwc', 'timecallable']


def constant(
    qarray: QArrayLike, *, tstart: float = -jnp.inf, tend: float = jnp.inf
) -> ConstantTimeQArray:
    r"""Instantiate a constant time-qarray.

    A constant time-qarray is defined by $O(t) = O_0$ for any time $t$, where $O_0$ is a
    constant qarray.

    Args:
        qarray _(qarray-like of shape (..., n, n))_: Constant qarray $O_0$.

    Returns:
        _(time-qarray of shape (..., n, n) when called)_ Callable returning $O_0$ for
            any time $t$.

    Examples:
        >>> H = dq.constant(dq.sigmaz())
        >>> H(0.0)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
        [[ 1.+0.j    ⋅   ]
         [   ⋅    -1.+0.j]]
        >>> H(1.0)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
        [[ 1.+0.j    ⋅   ]
         [   ⋅    -1.+0.j]]
    """
    qarray = asqarray(qarray)
    check_shape(qarray, 'qarray', '(..., n, n)')
    return ConstantTimeQArray(tstart, tend, qarray)


def pwc(times: ArrayLike, values: ArrayLike, qarray: QArrayLike) -> PWCTimeQArray:
    r"""Instantiate a piecewise constant (PWC) time-qarray.

    A PWC time-qarray takes constant values over some time intervals. It is defined by
    $$
        O(t) = \left(\sum_{k=0}^{N-1} c_k\; \Omega_{[t_k, t_{k+1}[}(t)\right) O_0
    $$
    where $c_k$ are constant values, $\Omega_{[t_k, t_{k+1}[}$ is the rectangular
    window function defined by $\Omega_{[t_a, t_b[}(t) = 1$ if $t \in [t_a, t_b[$ and
    $\Omega_{[t_a, t_b[}(t) = 0$ otherwise, and $O_0$ is a constant qarray.

    Note:
        The argument `times` must be sorted in ascending order, but does not
        need to be evenly spaced.

    Note:
        If the returned time-qarray is called for a time $t$ which does not belong to
        any time intervals, the returned qarray is null.

    Args:
        times _(array-like of shape (N+1,))_: Time points $t_k$ defining the boundaries
            of the time intervals, where _N_ is the number of time intervals.
        values _(array-like of shape (..., N))_: Constant values $c_k$ for each time
            interval.
        qarray _(qarray-like of shape (n, n))_: Constant qarray $O_0$.

    Returns:
        _(time-qarray of shape (..., n, n) when called)_ Callable returning $O(t)$ for
            any time $t$.

    Examples:
        >>> times = [0.0, 1.0, 2.0]
        >>> values = [3.0, -2.0]
        >>> qarray = dq.sigmaz()
        >>> H = dq.pwc(times, values, qarray)
        >>> H(-0.5)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
        [[  ⋅      ⋅   ]
         [  ⋅      ⋅   ]]
        >>> H(0.0)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
        [[ 3.+0.j    ⋅   ]
         [   ⋅    -3.+0.j]]
        >>> H(0.5)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
        [[ 3.+0.j    ⋅   ]
         [   ⋅    -3.+0.j]]
        >>> H(1.0)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
        [[-2.+0.j    ⋅   ]
         [   ⋅     2.+0.j]]
    """
    # times
    times = jnp.asarray(times)
    times = check_times(times, 'times')

    # values
    values = jnp.asarray(values, dtype=cdtype())
    if values.shape[-1] != len(times) - 1:
        raise TypeError(
            'Argument `values` must have shape `(..., len(times)-1)`, but has shape'
            f' `{values.shape}.'
        )

    # qarray
    qarray = asqarray(qarray)
    check_shape(qarray, 'qarray', '(n, n)')

    return PWCTimeQArray(times[0], times[-1], times, values, qarray)


def modulated(
    f: callable[[float], Scalar | Array],
    qarray: QArrayLike,
    *,
    tstart: float = -jnp.inf,
    tend: float = jnp.inf,
    discontinuity_ts: ArrayLike | None = None,
) -> ModulatedTimeQArray:
    r"""Instantiate a modulated time-qarray.

    A modulated time-qarray is defined by $O(t) = f(t) O_0$ where $f(t)$ is a
    time-dependent scalar. The function $f$ is defined by passing a Python function
    with signature `f(t: float) -> Scalar | Array` that returns a scalar or an array of
    shape _(...)_ for any time $t$.

    Args:
        f _(function returning scalar or array of shape (...))_: Function with signature
            `f(t: float) -> Scalar | Array` that returns the modulating factor
            $f(t)$.
        qarray _(qarray-like of shape (n, n))_: Constant qarray $O_0$.
        discontinuity_ts _(array-like, optional)_: Times at which there is a
            discontinuous jump in the function values.

    Returns:
        _(time-qarray of shape (..., n, n) when called)_ Callable returning $O(t)$ for
            any time $t$.

    Examples:
        >>> f = lambda t: jnp.cos(2.0 * jnp.pi * t)
        >>> H = dq.modulated(f, dq.sigmax())
        >>> H(0.5)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=2
        [[   ⋅    -1.+0.j]
         [-1.+0.j    ⋅   ]]
        >>> H(1.0)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=2
        [[  ⋅    1.+0.j]
         [1.+0.j   ⋅   ]]
    """
    # check f is callable
    if not callable(f):
        raise TypeError(
            f'Argument `f` must be a function, but has type {obj_type_str(f)}.'
        )

    # qarray
    qarray = asqarray(qarray)
    check_shape(qarray, 'qarray', '(n, n)')

    # discontinuity_ts
    if discontinuity_ts is not None:
        discontinuity_ts = jnp.asarray(discontinuity_ts)
        discontinuity_ts = jnp.sort(discontinuity_ts)

    # make f a valid PyTree that is vmap-compatible
    f = BatchedCallable(f)

    return ModulatedTimeQArray(tstart, tend, f, qarray, discontinuity_ts)


def timecallable(
    f: callable[[float], QArray],
    *,
    tstart: float = -jnp.inf,
    tend: float = jnp.inf,
    discontinuity_ts: ArrayLike | None = None,
) -> CallableTimeQArray:
    r"""Instantiate a callable time-qarray.

    A callable time-qarray is defined by $O(t) = f(t)$ where $f(t)$ is a
    time-dependent operator. The function $f$ is defined by passing a Python function
    with signature `f(t: float) -> QArray` that returns a qarray of shape _(..., n, n)_
    for any time $t$.

    Warning: The function `f` must return a qarray (not a qarray-like!)
        An error is raised if the function `f` does not return a qarray. This error
        concerns any other qarray-likes. This is enforced to avoid costly
        conversions at every time step of the numerical integration.

    Args:
        f _(function returning qarray of shape (..., n, n))_: Function with
            signature `(t: float) -> QArray` that returns the qarray $f(t)$.
        discontinuity_ts _(array-like, optional)_: Times at which there is a
            discontinuous jump in the function values.

    Returns:
        _(time-qarray of shape (..., n, n) when called)_ Callable returning $O(t)$ for
            any time $t$.

    Examples:
        >>> f = lambda t: dq.asqarray([[t, 0], [0, 1 - t]])
        >>> H = dq.timecallable(f)
        >>> H(0.5)
        QArray: shape=(2, 2), dims=(2,), dtype=float32, layout=dense
        [[0.5 0. ]
         [0.  0.5]]
        >>> H(1.0)
        QArray: shape=(2, 2), dims=(2,), dtype=float32, layout=dense
        [[1. 0.]
         [0. 0.]]
    """
    # check f is callable
    if not callable(f):
        raise TypeError(
            f'Argument `f` must be a function, but has type {obj_type_str(f)}.'
        )

    # discontinuity_ts
    if discontinuity_ts is not None:
        discontinuity_ts = jnp.asarray(discontinuity_ts)
        discontinuity_ts = jnp.sort(discontinuity_ts)

    # make f a valid PyTree that is vmap-compatible
    f = BatchedCallable(f)

    return CallableTimeQArray(tstart, tend, f, discontinuity_ts)


class Shape(tuple):
    """Helper class to help with Pytree handling."""


class TimeQArray(eqx.Module):
    r"""Base class for time-dependent qarrays.

    A time-qarray is a callable object that returns a qarray for any time $t$. It is
    used to define time-dependent operators for Dynamiqs solvers.

    Attributes:
        dtype _(numpy.dtype)_: Data type.
        shape _(tuple of int)_: Shape.
        ndim _(int)_: Number of dimensions in the shape.
        layout _(Layout)_: Data layout, either `dq.dense` or `dq.dia`.
        dims _(tuple of ints)_: Hilbert space dimension of each subsystem.
        mT _(time-qarray)_: Returns the time-qarray transposed over its last two
            dimensions.
        vectorized _(bool)_: Whether the underlying qarray is non-vectorized (ket, bra
            or operator) or vectorized (operator in vector form or superoperator in
            matrix form).
        discontinuity_ts _(Array | None)_: Times at which there is a discontinuous jump
            in the time-qarray values (the array is always sorted, but does not
            necessarily contain unique values).

    Note: Arithmetic operation support
        Time-qarrays support basic arithmetic operations `-, +, *` with other
        qarray-likes or time-qarrays.
    """

    # Subclasses should implement:
    # - the properties: dtype, shape, dims, ndiags, vectorized, layout, mT, in_axes,
    #                   discontinuity_ts
    # - the methods: reshape, broadcast_to, conj, _call, __mul__

    tstart: float
    tend: float

    def _replace(
        self, tstart: float | None = None, tend: float | None = None, **kwargs
    ) -> TimeQArray:
        if tstart is None:
            tstart = self.tstart
        if tend is None:
            tend = self.tend
        return type(self)(tstart=tstart, tend=tend, **kwargs)

    @property
    @abstractmethod
    def dtype(self) -> jnp.dtype:
        pass

    @property
    @abstractmethod
    def shape(self) -> tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def dims(self) -> tuple[int, ...]:
        pass

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    @abstractmethod
    def ndiags(self) -> int:
        pass

    @property
    @abstractmethod
    def vectorized(self) -> bool:
        pass

    @property
    @abstractmethod
    def layout(self) -> Layout:
        pass

    @property
    @abstractmethod
    def mT(self) -> TimeQArray:
        pass

    @property
    @abstractmethod
    def in_axes(self) -> PyTree[int | None]:
        # returns the `in_axes` arguments that should be passed to vmap in order
        # to vmap the `TimeQArray` correctly
        pass

    @property
    @abstractmethod
    def discontinuity_ts(self) -> Array | None:
        # must be sorted, not necessarily unique values
        pass

    @abstractmethod
    def reshape(self, *shape: int) -> TimeQArray:
        """Returns a reshaped copy of a time-qarray.

        Args:
            *shape: New shape, which must match the original size.

        Returns:
            New time-qarray with the given shape.
        """

    @abstractmethod
    def broadcast_to(self, *shape: int) -> TimeQArray:
        """Broadcasts a time-qarray to a new shape.

        Args:
            *shape: New shape, which must be compatible with the original shape.

        Returns:
            New time-qarray with the given shape.
        """

    @abstractmethod
    def conj(self) -> TimeQArray:
        """Returns the element-wise complex conjugate of the time-qarray.

        Returns:
            New time-qarray with element-wise complex conjuguated values.
        """

    def dag(self) -> TimeQArray:
        r"""Returns the adjoint (complex conjugate transpose) of the time-qarray.

        Returns:
            New time-qarray with adjoint values.
        """
        return self.mT.conj()

    def squeeze(self, axis: int | None = None) -> TimeQArray:
        """Squeeze a time-qarray.

        Args:
            axis: Axis to squeeze. If `none`, all axes with dimension 1 are squeezed.

        Returns:
            New time-qarray with squeezed_shape
        """
        if axis is None:
            shape = self.shape
            x = self
            for i, s in reversed(list(enumerate(shape))):
                if s == 1:
                    x = x.squeeze(i)
            return x

        if axis >= self.ndim:
            raise ValueError(
                f'Cannot squeeze axis {axis} from a time-qarray with {self.ndim} axes.'
            )
        return self.reshape(*self.shape[:axis], *self.shape[axis + 1 :])

    def _indicator(self, t: ScalarLike) -> float:
        return lax.cond(
            jnp.logical_or(t < self.tstart, t >= self.tend), lambda: 0, lambda: 1
        )

    def __call__(self, t: ScalarLike) -> QArray:
        """Returns the time-qarray evaluated at a given time.

        Args:
            t: Time at which to evaluate the time-qarray.

        Returns:
            Qarray evaluated at time $t$.
        """
        return self._indicator(t) * self._call(t)

    @abstractmethod
    def _call(self, t: ScalarLike) -> QArray:
        # tstart <= t < tend
        pass

    def __neg__(self) -> TimeQArray:
        return self * (-1)

    @abstractmethod
    def __mul__(self, y: QArrayLike) -> TimeQArray:
        pass

    def __rmul__(self, y: QArrayLike) -> TimeQArray:
        return self * y

    def __add__(self, y: QArrayLike | TimeQArray) -> TimeQArray:
        if isqarraylike(y):
            y = ConstantTimeQArray(-jnp.inf, jnp.inf, asqarray(y))

        if isinstance(y, TimeQArray):
            tstart = jnp.minimum(self.tstart, y.tstart)
            tend = jnp.maximum(self.tend, y.tend)
            return SummedTimeQArray(tstart, tend, [self, y])
        else:
            return NotImplemented

    def __radd__(self, y: QArrayLike | TimeQArray) -> TimeQArray:
        return self + y

    def __sub__(self, y: QArrayLike | TimeQArray) -> TimeQArray:
        return self + (-y)

    def __rsub__(self, y: QArrayLike | TimeQArray) -> TimeQArray:
        return y + (-self)

    def __repr__(self) -> str:
        res = (
            f'{type(self).__name__}: shape={self.shape}, dims={self.dims}, '
            f'dtype={self.dtype}, layout={self.layout}'
        )
        if self.vectorized:
            res += f', vectorized={self.vectorized}'
        if self.layout is dia:
            res += f', ndiags={self.ndiags}'
        return res


class ConstantTimeQArray(TimeQArray):
    qarray: QArray

    def _replace(self, qarray: QArray | None = None) -> TimeQArray:
        if qarray is None:
            qarray = self.qarray
        return super()._replace(qarray=qarray)

    @property
    def dtype(self) -> jnp.dtype:
        return self.qarray.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.qarray.shape

    @property
    def dims(self) -> tuple[int, ...]:
        return self.qarray.dims

    @property
    def ndiags(self) -> int:
        return self.qarray.ndiags

    @property
    def vectorized(self) -> bool:
        return self.qarray.vectorized

    @property
    def layout(self) -> Layout:
        return self.qarray.layout

    @property
    def mT(self) -> TimeQArray:
        qarray = self.qarray.mT
        return self._replace(qarray=qarray)

    @property
    def in_axes(self) -> PyTree[int | None]:
        return ConstantTimeQArray(None, None, 0)

    @property
    def discontinuity_ts(self) -> Array | None:
        return None

    def reshape(self, *shape: int) -> TimeQArray:
        qarray = self.qarray.reshape(*shape)
        return self._replace(qarray=qarray)

    def broadcast_to(self, *shape: int) -> TimeQArray:
        qarray = self.qarray.broadcast_to(*shape)
        return self._replace(qarray=qarray)

    def conj(self) -> TimeQArray:
        qarray = self.qarray.conj()
        return self._replace(qarray=qarray)

    def _call(self, t: ScalarLike) -> QArray:  # noqa: ARG002
        return self.qarray

    def __mul__(self, y: QArrayLike) -> TimeQArray:
        qarray = self.qarray * y
        return self._replace(qarray=qarray)

    def __add__(self, y: QArrayLike | TimeQArray) -> TimeQArray:
        # handle addition with a constant object as a special case
        if isqarraylike(y):
            y = ConstantTimeQArray(-jnp.inf, jnp.inf, asqarray(y))

        if isinstance(y, ConstantTimeQArray):
            tstart = jnp.minimum(self.tstart, y.tstart)
            tend = jnp.maximum(self.tend, y.tend)
            return ConstantTimeQArray(tstart, tend, self.qarray + y.qarray)

        return super().__add__(y)


class PWCTimeQArray(TimeQArray):
    times: Array  # (nv+1,)
    values: Array  # (..., nv)
    qarray: QArray  # (n, n)

    def _replace(
        self, values: Array | None = None, qarray: QArray | None = None
    ) -> TimeQArray:
        if values is None:
            values = self.values
        if qarray is None:
            qarray = self.qarray
        return super()._replace(times=self.times, values=values, qarray=qarray)

    @property
    def dtype(self) -> jnp.dtype:
        return self.qarray.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return *self.values.shape[:-1], *self.qarray.shape

    @property
    def dims(self) -> tuple[int, ...]:
        return self.qarray.dims

    @property
    def ndiags(self) -> int:
        return self.qarray.ndiags

    @property
    def vectorized(self) -> int:
        return self.qarray.vectorized

    @property
    def layout(self) -> Layout:
        return self.qarray.layout

    @property
    def mT(self) -> TimeQArray:
        qarray = self.qarray.mT
        return self._replace(qarray=qarray)

    @property
    def in_axes(self) -> PyTree[int | None]:
        return PWCTimeQArray(None, None, None, 0, None)

    @property
    def discontinuity_ts(self) -> Array | None:
        return self.times

    def reshape(self, *shape: int) -> TimeQArray:
        shape = shape[:-2] + self.values.shape[-1:]  # (..., nv)
        values = self.values.reshape(*shape)
        return self._replace(values=values)

    def broadcast_to(self, *shape: int) -> TimeQArray:
        shape = shape[:-2] + self.values.shape[-1:]  # (..., nv)
        values = jnp.broadcast_to(self.values, shape)
        return self._replace(values=values)

    def conj(self) -> TimeQArray:
        values = self.values.conj()
        qarray = self.qarray.conj()
        return self._replace(values=values, qarray=qarray)

    def _call(self, t: ScalarLike) -> QArray:
        idx = jnp.searchsorted(self.times, t, side='right') - 1
        prefactor = self.values[..., idx]  # (...)
        return prefactor[..., None, None] * self.qarray

    def __mul__(self, y: QArrayLike) -> TimeQArray:
        qarray = self.qarray * y
        return self._replace(qarray=qarray)


class ModulatedTimeQArray(TimeQArray):
    f: BatchedCallable  # (...)
    qarray: QArray  # (n, n)
    _disc_ts: Array | None

    def _replace(
        self, f: BatchedCallable | None = None, qarray: QArray | None = None
    ) -> TimeQArray:
        if f is None:
            f = self.f
        if qarray is None:
            qarray = self.qarray
        return super()._replace(f=f, qarray=qarray, _disc_ts=self._disc_ts)

    @property
    def dtype(self) -> jnp.dtype:
        return self.qarray.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return *self.f.shape, *self.qarray.shape

    @property
    def dims(self) -> tuple[int, ...]:
        return self.qarray.dims

    @property
    def ndiags(self) -> int:
        return self.qarray.ndiags

    @property
    def vectorized(self) -> bool:
        return self.qarray.vectorized

    @property
    def layout(self) -> Layout:
        return self.qarray.layout

    @property
    def mT(self) -> TimeQArray:
        qarray = self.qarray.mT
        return self._replace(qarray=qarray)

    @property
    def in_axes(self) -> PyTree[int | None]:
        return ModulatedTimeQArray(None, None, 0, None, None)

    @property
    def discontinuity_ts(self) -> Array | None:
        return self._disc_ts

    def reshape(self, *shape: int) -> TimeQArray:
        f = self.f.reshape(*shape[:-2])
        return self._replace(f=f)

    def broadcast_to(self, *shape: int) -> TimeQArray:
        f = self.f.broadcast_to(*shape[:-2])
        return self._replace(f=f)

    def conj(self) -> TimeQArray:
        f = self.f.conj()
        qarray = self.qarray.conj()
        return self._replace(f=f, qarray=qarray)

    def _call(self, t: ScalarLike) -> QArray:
        return self.f(t)[..., None, None] * self.qarray

    def __mul__(self, y: QArrayLike) -> TimeQArray:
        qarray = self.qarray * y
        return self._replace(qarray=qarray)


class CallableTimeQArray(TimeQArray):
    f: BatchedCallable  # (..., n, n)
    _disc_ts: Array | None

    def _replace(self, f: BatchedCallable | None = None) -> TimeQArray:
        if f is None:
            f = self.f
        return super()._replace(f=f, _disc_ts=self._disc_ts)

    @property
    def dtype(self) -> jnp.dtype:
        return self.f.dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return self.f.shape

    @property
    def dims(self) -> tuple[int, ...]:
        return jax.eval_shape(self.f, 0.0).dims

    @property
    def ndiags(self) -> int:
        return jax.eval_shape(self.f, 0.0).ndiags

    @property
    def vectorized(self) -> bool:
        return jax.eval_shape(self.f, 0.0).vectorized

    @property
    def layout(self) -> Layout:
        return self.f.layout

    @property
    def mT(self) -> TimeQArray:
        f = jtu.Partial(lambda t: self.f(t).mT)
        return self._replace(f=f)

    @property
    def in_axes(self) -> PyTree[int | None]:
        return CallableTimeQArray(None, None, 0, None)

    @property
    def discontinuity_ts(self) -> Array | None:
        return self._disc_ts

    def reshape(self, *shape: int) -> TimeQArray:
        f = self.f.reshape(*shape)
        return self._replace(f=f)

    def broadcast_to(self, *shape: int) -> TimeQArray:
        f = self.f.broadcast_to(*shape)
        return self._replace(f=f)

    def conj(self) -> TimeQArray:
        f = self.f.conj()
        return self._replace(f=f)

    def _call(self, t: ScalarLike) -> QArray:
        return self.f(t)

    def __mul__(self, y: QArrayLike) -> TimeQArray:
        f = self.f * y
        return self._replace(f=f)


class SummedTimeQArray(TimeQArray):
    timeqarrays: list[TimeQArray]

    def _replace(self, timeqarrays: list[TimeQArray] | None = None) -> TimeQArray:
        if timeqarrays is None:
            timeqarrays = self.timeqarrays
        return super()._replace(timeqarrays=timeqarrays)

    def __init__(self, tstart: float, tend: float, timeqarrays: list[TimeQArray]):
        super().__init__(tstart, tend)

        # dirty way to bypass the check for the implementation of `in_axes`
        if tstart is not None and tend is not None:
            # verify all time-qarrays of the sum are broadcast compatible
            shape = jnp.broadcast_shapes(*[tqarray.shape for tqarray in timeqarrays])
            # ensure all time-qarrays can be jointly vmapped over (as specified by the
            # `in_axes` property)
            timeqarrays = [tqarray.broadcast_to(*shape) for tqarray in timeqarrays]

            dims = {t.dims for t in timeqarrays}
            if len(dims) > 1:
                raise ValueError(
                    f'All terms of a SummedTimeArray must have the'
                    f'same Hilbert space dimensions, got {dims}'
                )

        self.timeqarrays = timeqarrays

    @property
    def dtype(self) -> jnp.dtype:
        dtypes = [tqarray.dtype for tqarray in self.timeqarrays]
        return ft.reduce(jnp.promote_types, dtypes)

    @property
    def shape(self) -> tuple[int, ...]:
        return jnp.broadcast_shapes(*[tqarray.shape for tqarray in self.timeqarrays])

    @property
    def dims(self) -> tuple[int, ...]:
        return self.timearrays[0].dims

    @property
    def ndiags(self) -> int:
        return jax.eval_shape(self.__call__, 0.0).ndiags

    @property
    def vectorized(self) -> bool:
        return jax.eval_shape(self.__call__, 0.0).vectorized

    @property
    def layout(self) -> Layout:
        layouts = [tqarray.layout for tqarray in self.timeqarrays]
        return ft.reduce(promote_layouts, layouts)

    @property
    def mT(self) -> TimeQArray:
        timeqarrays = [tqarray.mT for tqarray in self.timeqarrays]
        return self._replace(timeqarrays=timeqarrays)

    @property
    def in_axes(self) -> PyTree[int | None]:
        in_axes_list = [tqarray.in_axes for tqarray in self.timeqarrays]
        return SummedTimeQArray(None, None, in_axes_list)

    @property
    def discontinuity_ts(self) -> Array | None:
        ts = [tqarray.discontinuity_ts for tqarray in self.timeqarrays]
        return _concatenate_sort(*ts)

    def reshape(self, *shape: int) -> TimeQArray:
        timeqarrays = [tqarray.reshape(*shape) for tqarray in self.timeqarrays]
        return self._replace(timeqarrays=timeqarrays)

    def broadcast_to(self, *shape: int) -> TimeQArray:
        timeqarrays = [tqarray.broadcast_to(*shape) for tqarray in self.timeqarrays]
        return self._replace(timeqarrays=timeqarrays)

    def conj(self) -> TimeQArray:
        timeqarrays = [tqarray.conj() for tqarray in self.timeqarrays]
        return self._replace(timeqarrays=timeqarrays)

    def _call(self, t: ScalarLike) -> QArray:
        return ft.reduce(
            lambda x, y: x + y, [tqarray(t) for tqarray in self.timeqarrays]
        )

    def __mul__(self, y: QArrayLike) -> TimeQArray:
        timeqarrays = [tqarray * y for tqarray in self.timeqarrays]
        return self._replace(timeqarrays=timeqarrays)

    def __add__(self, y: QArrayLike | TimeQArray) -> TimeQArray:
        if isqarraylike(y):
            y = ConstantTimeQArray(-jnp.inf, jnp.inf, asqarray(y))

        tstart = jnp.minimum(self.tstart, y.tstart)
        tend = jnp.maximum(self.tend, y.tend)
        return SummedTimeQArray(tstart, tend, [*self.timeqarrays, y])


class BatchedCallable(eqx.Module):
    # this class turns a callable into a PyTree that is vmap-compatible

    f: callable[[float], QArrayLike]
    indices: list[Array]

    def __init__(self, f: callable[[float], QArrayLike]):
        # make f a valid PyTree with `Partial` and convert its output to a qarray
        self.f = jtu.Partial(f)
        eval_shape = jax.eval_shape(f, 0.0)
        if isinstance(eval_shape, QArray):
            shape = eval_shape.shape[:-2]
        else:
            shape = eval_shape.shape
        self.indices = list(jnp.indices(shape))

    def __call__(self, t: ScalarLike) -> QArrayLike:
        if len(self.indices) == 0:
            return self.f(t)
        else:
            return self.f(t)[tuple(self.indices)]

    @property
    def dtype(self) -> tuple[int, ...]:
        return jax.eval_shape(self.f, 0.0).dtype

    @property
    def shape(self) -> tuple[int, ...]:
        return jax.eval_shape(self.f, 0.0).shape

    @property
    def layout(self) -> Layout:
        return jax.eval_shape(self.f, 0.0).layout

    def reshape(self, *shape: tuple[int, ...]) -> BatchedCallable:
        f = lambda t: self.f(t).reshape(*shape)
        return BatchedCallable(f)

    def broadcast_to(self, *shape: tuple[int, ...]) -> BatchedCallable:
        def f(t: float) -> QArrayLike:
            res = self.f(t)
            if isinstance(res, QArray):
                return res.broadcast_to(*shape)
            else:
                return jnp.broadcast_to(res, shape)

        return BatchedCallable(f)

    def conj(self) -> BatchedCallable:
        f = lambda t: self.f(t).conj()
        return BatchedCallable(f)

    def squeeze(self, i: int) -> BatchedCallable:
        f = lambda t: jnp.squeeze(self.f(t), i)
        return BatchedCallable(f)

    def __add__(self, y: ScalarLike) -> BatchedCallable:
        return BatchedCallable(lambda t: self.f(t) + y)

    def __mul__(self, y: ArrayLike) -> BatchedCallable:
        f = lambda t: self.f(t) * y
        return BatchedCallable(f)
