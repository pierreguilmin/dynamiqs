from __future__ import annotations

from math import prod

import jax.numpy as jnp
from jax.typing import ArrayLike

from .._utils import cdtype
from ..qarrays.dense_qarray import DenseQArray
from ..qarrays.layout import Layout, dense, get_layout
from ..qarrays.qarray import QArray, QArrayLike, get_dims
from ..qarrays.sparsedia_qarray import SparseDIAQArray
from ..qarrays.utils import asqarray, init_dims, sparsedia_from_dict, to_jax
from .general import tensor

__all__ = [
    'cnot',
    'create',
    'destroy',
    'displace',
    'eye',
    'eye_like',
    'hadamard',
    'momentum',
    'number',
    'parity',
    'position',
    'quadrature',
    'rx',
    'ry',
    'rz',
    'sgate',
    'sigmam',
    'sigmap',
    'sigmax',
    'sigmay',
    'sigmaz',
    'squeeze',
    'tgate',
    'toffoli',
    'zeros',
    'zeros_like',
]


def eye(*dims: int, layout: Layout | None = None) -> QArray:
    r"""Returns the identity operator.

    If multiple dimensions are provided $\mathtt{dims}=(n_1,\dots,n_N)$, it returns the
    identity operator of the composite Hilbert space of dimension $n=\prod n_k$:
    $$
        I_n = I_{n_1}\otimes\dots\otimes I_{n_N}.
    $$

    Args:
        *dims: Hilbert space dimension of each subsystem.
        layout: Matrix layout (`dq.dense`, `dq.dia` or `None`).

    Returns:
        _(qarray of shape (n, n))_ Identity operator, with _n = prod(dims)_.

    Examples:
        Single-mode $I_4$:
        >>> dq.eye(4)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dia, ndiags=1
        [[1.+0.j   ⋅      ⋅      ⋅   ]
         [  ⋅    1.+0.j   ⋅      ⋅   ]
         [  ⋅      ⋅    1.+0.j   ⋅   ]
         [  ⋅      ⋅      ⋅    1.+0.j]]

        Multi-mode $I_2 \otimes I_3$:
        >>> dq.eye(2, 3)
        QArray: shape=(6, 6), dims=(2, 3), dtype=complex64, layout=dia, ndiags=1
        [[1.+0.j   ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅    1.+0.j   ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅    1.+0.j   ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅    1.+0.j   ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅    1.+0.j   ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅    1.+0.j]]

    See also:
        - [`dq.eye_like()`][dynamiqs.eye_like]: returns the identity operator in the
        Hilbert space of the input.
    """
    layout = get_layout(layout)
    dim = prod(dims)
    if layout is dense:
        array = jnp.eye(dim, dtype=cdtype())
        return asqarray(array, dims=dims)
    else:
        diag = jnp.ones(dim, dtype=cdtype())
        return sparsedia_from_dict({0: diag}, dims=dims)


def eye_like(
    x: QArrayLike, dims: tuple[int, ...] | None = None, layout: Layout | None = None
) -> QArray:
    r"""Returns the identity operator in the Hilbert space of the input.

    Args:
        x _(qarray-like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra
            or operator.
        dims _(tuple of ints or None)_: Dimensions of each subsystem in the composite
            system Hilbert space tensor product. Defaults to `None` (`x.dims` if
            available, single Hilbert space `dims=(n,)` otherwise).
        layout _(dq.dense, dq.dia or None)_: Matrix layout.

    Returns:
        _(qarray of shape (n, n))_ Identity operator, with _n = prod(dims)_.

    Examples:
        Single-mode $I_4$:
        >>> a = dq.destroy(4)
        >>> dq.eye_like(a)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dia, ndiags=1
        [[1.+0.j   ⋅      ⋅      ⋅   ]
         [  ⋅    1.+0.j   ⋅      ⋅   ]
         [  ⋅      ⋅    1.+0.j   ⋅   ]
         [  ⋅      ⋅      ⋅    1.+0.j]]

        Multi-mode $I_2 \otimes I_3$:
        >>> a, b = dq.destroy(2, 3)
        >>> dq.eye_like(a)
        QArray: shape=(6, 6), dims=(2, 3), dtype=complex64, layout=dia, ndiags=1
        [[1.+0.j   ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅    1.+0.j   ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅    1.+0.j   ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅    1.+0.j   ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅    1.+0.j   ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅    1.+0.j]]

    See also:
        - [`dq.eye()`][dynamiqs.eye]: returns the identity operator.
    """
    xdims = get_dims(x)
    # todo: we should rather use a _get_shape util that never converts to a jax array
    x = to_jax(x)
    dims = init_dims(xdims, dims, x.shape)
    return eye(*dims, layout=layout)


def zeros(*dims: int, layout: Layout | None = None) -> QArray:
    r"""Returns the null operator.

    If multiple dimensions are provided $\mathtt{dims}=(n_1,\dots,n_N)$, it returns the
    null operator of the composite Hilbert space of dimension $n=\prod n_k$:
    $$
        0_n = 0_{n_1}\otimes\dots\otimes 0_{n_N}.
    $$

    Args:
        *dims: Hilbert space dimension of each subsystem.
        layout: Matrix layout (`dq.dense`, `dq.dia` or `None`).

    Returns:
        _(qarray of shape (n, n))_ Null operator, with _n = prod(dims)_.

    Examples:
        Single-mode $0_4$:
        >>> dq.zeros(4)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dia, ndiags=0
        [[  ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅   ]]

        Multi-mode $0_2 \otimes 0_3$:
        >>> dq.zeros(2, 3)
        QArray: shape=(6, 6), dims=(2, 3), dtype=complex64, layout=dia, ndiags=0
        [[  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]]

    See also:
        - [`dq.zeros_like()`][dynamiqs.zeros_like]: returns the null operator in the
        Hilbert space of the input.
    """
    layout = get_layout(layout)
    dim = prod(dims)
    if layout is dense:
        array = jnp.zeros((dim, dim), dtype=cdtype())
        return asqarray(array, dims=dims)
    else:
        diags = jnp.zeros((0, dim), dtype=cdtype())
        return SparseDIAQArray(dims, False, (), diags)


def zeros_like(
    x: QArrayLike, dims: tuple[int, ...] | None = None, layout: Layout | None = None
) -> QArray:
    r"""Returns the null operator in the Hilbert space of the input.

    Args:
        x _(qarray-like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra
            or operator.
        dims _(tuple of ints or None)_: Dimensions of each subsystem in the composite
            system Hilbert space tensor product. Defaults to `None` (`x.dims` if
            available, single Hilbert space `dims=(n,)` otherwise).
        layout _(dq.dense, dq.dia or None)_: Matrix layout.

    Returns:
        _(qarray of shape (n, n))_ Null operator, with _n = prod(dims)_.

    Examples:
        Single-mode $0_4$:
        >>> a = dq.destroy(4)
        >>> dq.zeros_like(a)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dia, ndiags=0
        [[  ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅   ]]

        Multi-mode $0_2 \otimes 0_3$:
        >>> a, b = dq.destroy(2, 3)
        >>> dq.zeros_like(a)
        QArray: shape=(6, 6), dims=(2, 3), dtype=complex64, layout=dia, ndiags=0
        [[  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]]

    See also:
        - [`dq.zeros()`][dynamiqs.zeros]: returns the null operator.
    """
    xdims = get_dims(x)
    # todo: we should rather use a _get_shape util that never converts to a jax array
    x = to_jax(x)
    dims = init_dims(xdims, dims, x.shape)
    return zeros(*dims, layout=layout)


def destroy(*dims: int, layout: Layout | None = None) -> QArray | tuple[QArray, ...]:
    r"""Returns a bosonic annihilation operator, or a tuple of annihilation operators
    for a multi-mode system.

    If multiple dimensions are provided $\mathtt{dims}=(n_1,\dots,n_N)$, it returns a
    tuple with _len(dims)_ operators $(A_1,\dots,A_N)$, where $A_k$ is the annihilation
    operator acting on the $k$-th subsystem within the composite Hilbert space of
    dimension $n=\prod n_k$:
    $$
        A_k = I_{n_1} \otimes\dots\otimes a_{n_k} \otimes\dots\otimes I_{n_N}.
    $$

    Args:
        *dims: Hilbert space dimension of each mode.
        layout: Matrix layout (`dq.dense`, `dq.dia` or `None`).

    Returns:
        _(qarray or tuple of qarrays, each of shape (n, n))_ Annihilation operator(s),
            with _n = prod(dims)_.

    Examples:
        Single-mode $a$:
        >>> dq.destroy(4)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dia, ndiags=1
        [[    ⋅     1.   +0.j     ⋅         ⋅    ]
         [    ⋅         ⋅     1.414+0.j     ⋅    ]
         [    ⋅         ⋅         ⋅     1.732+0.j]
         [    ⋅         ⋅         ⋅         ⋅    ]]

        Multi-mode $a\otimes I_3$ and $I_2\otimes b$:
        >>> a, b = dq.destroy(2, 3)
        >>> a
        QArray: shape=(6, 6), dims=(2, 3), dtype=complex64, layout=dia, ndiags=1
        [[  ⋅      ⋅      ⋅    1.+0.j   ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅    1.+0.j   ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅    1.+0.j]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]]
        >>> b
        QArray: shape=(6, 6), dims=(2, 3), dtype=complex64, layout=dia, ndiags=1
        [[    ⋅     1.   +0.j     ⋅         ⋅         ⋅         ⋅    ]
         [    ⋅         ⋅     1.414+0.j     ⋅         ⋅         ⋅    ]
         [    ⋅         ⋅         ⋅         ⋅         ⋅         ⋅    ]
         [    ⋅         ⋅         ⋅         ⋅     1.   +0.j     ⋅    ]
         [    ⋅         ⋅         ⋅         ⋅         ⋅     1.414+0.j]
         [    ⋅         ⋅         ⋅         ⋅         ⋅         ⋅    ]]
    """
    layout = get_layout(layout)

    def destroy_single(dim: int) -> QArray:
        diag = jnp.sqrt(jnp.arange(1, dim, dtype=cdtype()))
        if layout is dense:
            return asqarray(jnp.diag(diag, k=1))
        else:
            return sparsedia_from_dict({1: diag})

    if len(dims) == 1:
        return destroy_single(dims[0])

    a = [destroy_single(dim) for dim in dims]
    Id = [eye(dim, layout=layout) for dim in dims]
    return tuple(
        tensor(*[a[j] if i == j else Id[j] for j in range(len(dims))])
        for i in range(len(dims))
    )


def create(*dims: int, layout: Layout | None = None) -> QArray | tuple[QArray, ...]:
    r"""Returns a bosonic creation operator, or a tuple of creation operators for a
    multi-mode system.

    If multiple dimensions are provided $\mathtt{dims}=(n_1,\dots,n_N)$, it returns a
    tuple with _len(dims)_ operators $(A_1^\dag,\dots,A_N^\dag)$, where $A_k^\dag$ is
    the creation operator acting on the $k$-th subsystem within the composite Hilbert
    space of dimension $n=\prod n_k$:
    $$
        A_k^\dag = I_{n_1} \otimes\dots\otimes a_{n_k}^\dag \otimes\dots\otimes I_{n_N}.
    $$

    Args:
        *dims: Hilbert space dimension of each mode.
        layout: Matrix layout (`dq.dense`, `dq.dia` or `None`).

    Returns:
        _(qarray or tuple of qarrays, each of shape (n, n))_ Creation operator(s), with
            _n = prod(dims)_.

    Examples:
        Single-mode $a^\dag$:
        >>> dq.create(4)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dia, ndiags=1
        [[    ⋅         ⋅         ⋅         ⋅    ]
         [1.   +0.j     ⋅         ⋅         ⋅    ]
         [    ⋅     1.414+0.j     ⋅         ⋅    ]
         [    ⋅         ⋅     1.732+0.j     ⋅    ]]

        Multi-mode $a^\dag\otimes I_3$ and $I_2\otimes b^\dag$:
        >>> adag, bdag = dq.create(2, 3)
        >>> adag
        QArray: shape=(6, 6), dims=(2, 3), dtype=complex64, layout=dia, ndiags=1
        [[  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [1.+0.j   ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅    1.+0.j   ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅    1.+0.j   ⋅      ⋅      ⋅   ]]
        >>> bdag
        QArray: shape=(6, 6), dims=(2, 3), dtype=complex64, layout=dia, ndiags=1
        [[    ⋅         ⋅         ⋅         ⋅         ⋅         ⋅    ]
         [1.   +0.j     ⋅         ⋅         ⋅         ⋅         ⋅    ]
         [    ⋅     1.414+0.j     ⋅         ⋅         ⋅         ⋅    ]
         [    ⋅         ⋅         ⋅         ⋅         ⋅         ⋅    ]
         [    ⋅         ⋅         ⋅     1.   +0.j     ⋅         ⋅    ]
         [    ⋅         ⋅         ⋅         ⋅     1.414+0.j     ⋅    ]]
    """
    layout = get_layout(layout)

    def create_single(dim: int) -> QArray:
        diag = jnp.sqrt(jnp.arange(1, dim, dtype=cdtype()))
        if layout is dense:
            return asqarray(jnp.diag(diag, k=-1))
        else:
            return sparsedia_from_dict({-1: diag})

    if len(dims) == 1:
        return create_single(dims[0])

    adag = [create_single(dim) for dim in dims]
    Id = [eye(dim, layout=layout) for dim in dims]
    return tuple(
        tensor(*[adag[j] if i == j else Id[j] for j in range(len(dims))])
        for i in range(len(dims))
    )


def number(*dims: int, layout: Layout | None = None) -> QArray | tuple[QArray, ...]:
    r"""Returns the number operator of a bosonic mode, or a tuple of number operators
    for a multi-mode system.

    For a single mode, it is defined by $N = a^\dag a$, where $a$ and $a^\dag$ are the
    mode annihilation and creation operators, respectively. If multiple dimensions are
    provided $\mathtt{dims}=(n_1,\dots,n_M)$, it returns a tuple with _len(dims)_
    operators $(N_1,\dots,N_M)$, where $N_k$ is the number operator acting on the $k$-th
    subsystem within the composite Hilbert space of dimension $n=\prod n_k$:
    $$
        N_k = I_{n_1} \otimes\dots\otimes a_{n_k}^\dag a_{n_k} \otimes\dots\otimes I_{n_M}.
    $$

    Args:
        *dims: Hilbert space dimension of each mode.
        layout: Matrix layout (`dq.dense`, `dq.dia` or `None`).

    Returns:
        _(qarray or tuple of qarrays, each of shape (n, n))_ Number operator(s), with
            _n = prod(dims)_.

    Examples:
        Single-mode $a^\dag a$:
        >>> dq.number(4)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dia, ndiags=1
        [[  ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅    1.+0.j   ⋅      ⋅   ]
         [  ⋅      ⋅    2.+0.j   ⋅   ]
         [  ⋅      ⋅      ⋅    3.+0.j]]

        Multi-mode $a^\dag a \otimes I_3$ and $I_2\otimes b^\dag b$:
        >>> na, nb = dq.number(2, 3)
        >>> na
        QArray: shape=(6, 6), dims=(2, 3), dtype=complex64, layout=dia, ndiags=1
        [[  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅    1.+0.j   ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅    1.+0.j   ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅    1.+0.j]]
        >>> nb
        QArray: shape=(6, 6), dims=(2, 3), dtype=complex64, layout=dia, ndiags=1
        [[  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅    1.+0.j   ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅    2.+0.j   ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅      ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅    1.+0.j   ⋅   ]
         [  ⋅      ⋅      ⋅      ⋅      ⋅    2.+0.j]]
    """  # noqa: E501
    layout = get_layout(layout)

    def number_single(dim: int) -> QArray:
        diag = jnp.arange(0, dim, dtype=cdtype())
        if layout is dense:
            return asqarray(jnp.diag(diag))
        else:
            return sparsedia_from_dict({0: diag})

    if len(dims) == 1:
        return number_single(dims[0])

    nums = [number_single(dim) for dim in dims]
    Id = [eye(dim, layout=layout) for dim in dims]
    return tuple(
        tensor(*[nums[j] if i == j else Id[j] for j in range(len(dims))])
        for i in range(len(dims))
    )


def parity(dim: int, *, layout: Layout | None = None) -> QArray:
    r"""Returns the parity operator of a bosonic mode.

    It is defined by $P = e^{i\pi a^\dag a}$, where $a$ and $a^\dag$ are the
    annihilation and creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.
        layout: Matrix layout (`dq.dense`, `dq.dia` or `None`).

    Returns:
        _(qarray of shape (dim, dim))_ Parity operator.

    Examples:
        >>> dq.parity(4)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dia, ndiags=1
        [[ 1.+0.j    ⋅       ⋅       ⋅   ]
         [   ⋅    -1.+0.j    ⋅       ⋅   ]
         [   ⋅       ⋅     1.+0.j    ⋅   ]
         [   ⋅       ⋅       ⋅    -1.+0.j]]
    """
    layout = get_layout(layout)
    diag = jnp.ones(dim, dtype=cdtype()).at[1::2].set(-1)
    if layout is dense:
        return asqarray(jnp.diag(diag))
    else:
        return sparsedia_from_dict({0: diag})


def displace(dim: int, alpha: ArrayLike) -> DenseQArray:
    r"""Returns the displacement operator of complex amplitude $\alpha$.

    It is defined by
    $$
        D(\alpha) = \exp(\alpha a^\dag - \alpha^* a),
    $$
    where $a$ and $a^\dag$ are the annihilation and creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.
        alpha _(array-like of shape (...))_: Displacement amplitude.

    Returns:
        _(qarray of shape (..., dim, dim))_ Displacement operator.

    Examples:
        >>> dq.displace(4, 0.5)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dense
        [[ 0.882+0.j -0.441+0.j  0.156+0.j -0.047+0.j]
         [ 0.441+0.j  0.662+0.j -0.542+0.j  0.27 +0.j]
         [ 0.156+0.j  0.542+0.j  0.442+0.j -0.697+0.j]
         [ 0.047+0.j  0.27 +0.j  0.697+0.j  0.662+0.j]]
        >>> dq.displace(4, [0.1, 0.2]).shape
        (2, 4, 4)
    """
    alpha = jnp.asarray(alpha, dtype=cdtype())
    alpha = alpha[..., None, None]  # (..., 1, 1)
    a = destroy(dim, layout=dense)  # (n, n)
    return (alpha * a.dag() - alpha.conj() * a).expm()


def squeeze(dim: int, z: ArrayLike) -> DenseQArray:
    r"""Returns the squeezing operator of complex squeezing amplitude $z$.

    It is defined by
    $$
        S(z) = \exp\left(\frac{1}{2}\left(z^* a^2 - z a^{\dag 2}\right)\right),
    $$
    where $a$ and $a^\dag$ are the annihilation and creation operators, respectively.

    Args:
        dim: Dimension of the Hilbert space.
        z _(array-like of shape (...))_: Squeezing amplitude.

    Returns:
        _(qarray of shape (..., dim, dim))_ Squeezing operator.

    Examples:
        >>> dq.squeeze(4, 0.5)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dense
        [[ 0.938+0.j  0.   +0.j  0.346+0.j  0.   +0.j]
         [ 0.   +0.j  0.818+0.j  0.   +0.j  0.575+0.j]
         [-0.346+0.j  0.   +0.j  0.938+0.j  0.   +0.j]
         [ 0.   +0.j -0.575+0.j  0.   +0.j  0.818+0.j]]
        >>> dq.squeeze(4, [0.1, 0.2]).shape
        (2, 4, 4)
    """
    z = jnp.asarray(z, dtype=cdtype())
    z = z[..., None, None]  # (..., 1, 1)
    a = destroy(dim, layout=dense)  # (n, n)
    a2 = a @ a
    return (0.5 * (z.conj() * a2 - z * a2.dag())).expm()


def quadrature(dim: int, phi: float, *, layout: Layout | None = None) -> QArray:
    r"""Returns the quadrature operator of phase angle $\phi$.

    It is defined by $x_\phi = (e^{i\phi} a^\dag + e^{-i\phi} a) / 2$, where $a$ and
    $a^\dag$ are the annihilation and creation operators respectively.

    Args:
        dim: Dimension of the Hilbert space.
        phi: Phase angle.
        layout: Matrix layout (`dq.dense`, `dq.dia` or `None`).

    Returns:
        _(qarray of shape (dim, dim))_ Quadrature operator.

    Examples:
        >>> dq.quadrature(3, 0.0)
        QArray: shape=(3, 3), dims=(3,), dtype=complex64, layout=dia, ndiags=2
        [[    ⋅     0.5  +0.j     ⋅    ]
         [0.5  +0.j     ⋅     0.707+0.j]
         [    ⋅     0.707+0.j     ⋅    ]]
        >>> dq.quadrature(3, jnp.pi / 2)
        QArray: shape=(3, 3), dims=(3,), dtype=complex64, layout=dia, ndiags=2
        [[   ⋅       -0.-0.5j      ⋅      ]
         [-0.+0.5j      ⋅       -0.-0.707j]
         [   ⋅       -0.+0.707j    ⋅      ]]
    """
    a = destroy(dim, layout=layout)
    return 0.5 * (jnp.exp(1j * phi) * a.dag() + jnp.exp(-1j * phi) * a)


def position(dim: int, *, layout: Layout | None = None) -> QArray:
    r"""Returns the position operator $x = (a^\dag + a) / 2$.

    Args:
        dim: Dimension of the Hilbert space.
        layout: Matrix layout (`dq.dense`, `dq.dia` or `None`).

    Returns:
        _(qarray of shape (dim, dim))_ Position operator.

    Examples:
        >>> dq.position(3)
        QArray: shape=(3, 3), dims=(3,), dtype=complex64, layout=dia, ndiags=2
        [[    ⋅     0.5  +0.j     ⋅    ]
         [0.5  +0.j     ⋅     0.707+0.j]
         [    ⋅     0.707+0.j     ⋅    ]]
    """
    a = destroy(dim, layout=layout)
    return 0.5 * (a + a.dag())


def momentum(dim: int, *, layout: Layout | None = None) -> QArray:
    r"""Returns the momentum operator $p = i (a^\dag - a) / 2$.

    Args:
        dim: Dimension of the Hilbert space.
        layout: Matrix layout (`dq.dense`, `dq.dia` or `None`).

    Returns:
        _(qarray of shape (dim, dim))_ Momentum operator.

    Examples:
        >>> dq.momentum(3)
        QArray: shape=(3, 3), dims=(3,), dtype=complex64, layout=dia, ndiags=2
        [[  ⋅       0.-0.5j     ⋅      ]
         [0.+0.5j     ⋅       0.-0.707j]
         [  ⋅       0.+0.707j   ⋅      ]]
    """
    a = destroy(dim, layout=layout)
    return 0.5j * (a.dag() - a)


def sigmax(*, layout: Layout | None = None) -> QArray:
    r"""Returns the Pauli $\sigma_x$ operator.

    It is defined by $\sigma_x = \begin{pmatrix} 0 & 1 \\ 1 & 0 \end{pmatrix}$.

    Args:
        layout: Matrix layout (`dq.dense`, `dq.dia` or `None`).

    Returns:
        _(qarray of shape (2, 2))_ Pauli $\sigma_x$ operator.

    Examples:
        >>> dq.sigmax()
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=2
        [[  ⋅    1.+0.j]
         [1.+0.j   ⋅   ]]
    """
    layout = get_layout(layout)
    if layout is dense:
        array = jnp.array([[0, 1], [1, 0]], dtype=cdtype())
        return asqarray(array)
    else:
        return sparsedia_from_dict({-1: [1], 1: [1]}, dtype=cdtype())


def sigmay(*, layout: Layout | None = None) -> QArray:
    r"""Returns the Pauli $\sigma_y$ operator.

    It is defined by $\sigma_y = \begin{pmatrix} 0 & -i \\ i & 0 \end{pmatrix}$.

    Args:
        layout: Matrix layout (`dq.dense`, `dq.dia` or `None`).

    Returns:
        _(qarray of shape (2, 2))_ Pauli $\sigma_y$ operator.

    Examples:
        >>> dq.sigmay()
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=2
        [[  ⋅    0.-1.j]
         [0.+1.j   ⋅   ]]
    """
    layout = get_layout(layout)
    if layout is dense:
        array = jnp.array([[0, -1j], [1j, 0]], dtype=cdtype())
        return asqarray(array)
    else:
        return sparsedia_from_dict({-1: [1j], 1: [-1j]}, dtype=cdtype())


def sigmaz(*, layout: Layout | None = None) -> QArray:
    r"""Returns the Pauli $\sigma_z$ operator.

    It is defined by $\sigma_z = \begin{pmatrix} 1 & 0 \\ 0 & -1 \end{pmatrix}$.

    Args:
        layout: Matrix layout (`dq.dense`, `dq.dia` or `None`).

    Returns:
        _(qarray of shape (2, 2))_ Pauli $\sigma_z$ operator.

    Examples:
        >>> dq.sigmaz()
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
        [[ 1.+0.j    ⋅   ]
         [   ⋅    -1.+0.j]]
    """
    layout = get_layout(layout)
    if layout is dense:
        array = jnp.array([[1, 0], [0, -1]], dtype=cdtype())
        return asqarray(array)
    else:
        return sparsedia_from_dict({0: [1, -1]}, dtype=cdtype())


def sigmap(*, layout: Layout | None = None) -> QArray:
    r"""Returns the Pauli raising operator $\sigma_+$.

    It is defined by $\sigma_+ = \begin{pmatrix} 0 & 1 \\ 0 & 0 \end{pmatrix}$.

    Args:
        layout: Matrix layout (`dq.dense`, `dq.dia` or `None`).

    Returns:
        _(qarray of shape (2, 2))_ Pauli $\sigma_+$ operator.

    Examples:
        >>> dq.sigmap()
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
        [[  ⋅    1.+0.j]
         [  ⋅      ⋅   ]]
    """
    layout = get_layout(layout)

    if layout is dense:
        array = jnp.array([[0, 1], [0, 0]], dtype=cdtype())
        return asqarray(array)
    else:
        return sparsedia_from_dict({1: [1]}, dtype=cdtype())


def sigmam(*, layout: Layout | None = None) -> QArray:
    r"""Returns the Pauli lowering operator $\sigma_-$.

    It is defined by $\sigma_- = \begin{pmatrix} 0 & 0 \\ 1 & 0 \end{pmatrix}$.

    Args:
        layout: Matrix layout (`dq.dense`, `dq.dia` or `None`).

    Returns:
        _(qarray of shape (2, 2))_ Pauli $\sigma_-$ operator.

    Examples:
        >>> dq.sigmam()
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
        [[  ⋅      ⋅   ]
         [1.+0.j   ⋅   ]]
    """
    layout = get_layout(layout)

    if layout is dense:
        array = jnp.array([[0, 0], [1, 0]], dtype=cdtype())
        return asqarray(array)
    else:
        return sparsedia_from_dict({-1: [1]}, dtype=cdtype())


def hadamard(n: int = 1) -> QArray:
    r"""Returns the Hadamard transform on $n$ qubits.

    For a single qubit, it is defined by
    $$
        H = \frac{1}{\sqrt2} \begin{pmatrix}
            1 & 1 \\\\
            1 & -1
        \end{pmatrix}
    $$
    For $n$ qubits, it is defined by the tensor product of Hadamard matrices:
    $$
        H_n = \bigotimes_{k=1}^n H
    $$

    Args:
        n: Number of qubits to act on.

    Returns:
        _(qarray of shape (2^n, 2^n))_ Hadamard transform operator.

    Examples:
        >>> dq.hadamard()
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[ 0.707+0.j  0.707+0.j]
         [ 0.707+0.j -0.707+0.j]]
        >>> dq.hadamard(2)
        QArray: shape=(4, 4), dims=(2, 2), dtype=complex64, layout=dense
        [[ 0.5+0.j  0.5+0.j  0.5+0.j  0.5+0.j]
         [ 0.5+0.j -0.5+0.j  0.5+0.j -0.5+0.j]
         [ 0.5+0.j  0.5+0.j -0.5+0.j -0.5+0.j]
         [ 0.5+0.j -0.5+0.j -0.5+0.j  0.5-0.j]]
    """
    H1 = jnp.array([[1, 1], [1, -1]], dtype=cdtype()) / jnp.sqrt(2)
    Hs = jnp.broadcast_to(H1, (n, 2, 2))  # (n, 2, 2)
    return tensor(*Hs)


def rx(theta: ArrayLike) -> QArray:
    r"""Returns the $R_x(\theta)$ rotation gate.

    It is defined by
    $$
        R_x(\theta) = \begin{pmatrix}
            \cos(\theta/2)   & -i\sin(\theta/2) \\\\
            -i\sin(\theta/2) & \cos(\theta/2)
        \end{pmatrix}
    $$

    Args:
        theta _(array-like of shape (...))_: Rotation angle $\theta$ in radians.

    Returns:
        _(qarray of shape (2, 2))_ $R_x(\theta)$ gate.

    Examples:
        >>> dq.rx(jnp.pi)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[-0.+0.j  0.-1.j]
         [ 0.-1.j -0.+0.j]]
        >>> dq.rx([0, jnp.pi/4, jnp.pi/3, jnp.pi/2, jnp.pi]).shape
        (5, 2, 2)
    """
    theta = jnp.asarray(theta)
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    rx = jnp.array([[c, -1j * s], [-1j * s, c]], dtype=cdtype())
    rx = jnp.moveaxis(rx, (0, 1), (-2, -1))
    return asqarray(rx)


def ry(theta: ArrayLike) -> QArray:
    r"""Returns the $R_y(\theta)$ rotation gate.

    It is defined by
    $$
        R_y(\theta) = \begin{pmatrix}
            \cos(\theta/2) & -\sin(\theta/2) \\\\
            \sin(\theta/2) & \cos(\theta/2)
        \end{pmatrix}
    $$

    Args:
        theta _(array-like of shape (...))_: Rotation angle $\theta$ in radians.

    Returns:
        _(qarray of shape (2, 2))_ $R_y(\theta)$ gate.

    Examples:
        >>> dq.ry(jnp.pi)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[-0.+0.j -1.+0.j]
         [ 1.+0.j -0.+0.j]]
        >>> dq.ry([0, jnp.pi/4, jnp.pi/3, jnp.pi/2, jnp.pi]).shape
        (5, 2, 2)
    """
    theta = jnp.asarray(theta)
    c = jnp.cos(theta / 2)
    s = jnp.sin(theta / 2)
    ry = jnp.array([[c, -s], [s, c]], dtype=cdtype())
    ry = jnp.moveaxis(ry, (0, 1), (-2, -1))
    return asqarray(ry)


def rz(theta: ArrayLike) -> QArray:
    r"""Returns the $R_z(\theta)$ rotation gate.

    It is defined by
    $$
        R_z(\theta) = \begin{pmatrix}
            e^{-i\theta/2} & 0 \\\\
            0              & e^{i\theta/2}
        \end{pmatrix}
    $$

    Args:
        theta _(array-like of shape (...))_: Rotation angle $\theta$ in radians.

    Returns:
        _(qarray of shape (2, 2))_ $R_z(\theta)$ gate.

    Examples:
        >>> dq.rz(jnp.pi)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[-0.-1.j  0.+0.j]
         [ 0.+0.j -0.+1.j]]
        >>> dq.rz([0, jnp.pi/4, jnp.pi/3, jnp.pi/2, jnp.pi]).shape
        (5, 2, 2)
    """
    theta = jnp.asarray(theta)
    zero = jnp.zeros_like(theta)
    rz = jnp.array(
        [[jnp.exp(-1j * theta / 2), zero], [zero, jnp.exp(1j * theta / 2)]],
        dtype=cdtype(),
    )
    rz = jnp.moveaxis(rz, (0, 1), (-2, -1))
    return asqarray(rz)


def sgate() -> QArray:
    r"""Returns the $\text{S}$ gate.

    It is defined by $\text{S} = \begin{pmatrix} 1 & 0 \\ 0 & i \end{pmatrix}$.

    Returns:
        _(qarray of shape (2, 2))_ $\text{S}$ gate.

    Examples:
        >>> dq.sgate()
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[1.+0.j 0.+0.j]
         [0.+0.j 0.+1.j]]
    """
    array = jnp.array([[1, 0], [0, 1j]], dtype=cdtype())
    return asqarray(array)


def tgate() -> QArray:
    r"""Returns the $\text{T}$ gate.

    It is defined by
    $\text{T} = \begin{pmatrix} 1 & 0 \\ 0 & e^{i\frac{\pi}{4}} \end{pmatrix}$.

    Returns:
        _(qarray of shape (2, 2))_ $\text{T}$ gate.

    Examples:
        >>> dq.tgate()
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[1.   +0.j    0.   +0.j   ]
         [0.   +0.j    0.707+0.707j]]
    """
    array = jnp.array([[1, 0], [0, (1 + 1j) / jnp.sqrt(2)]], dtype=cdtype())
    return asqarray(array)


def cnot() -> QArray:
    r"""Returns the $\text{CNOT}$ gate.

    It is defined by
    $$
        \text{CNOT} = \begin{pmatrix}
            1 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 \\\\
            0 & 0 & 0 & 1 \\\\
            0 & 0 & 1 & 0
        \end{pmatrix}
    $$

    Returns:
        _(qarray of shape (4, 4))_ $\text{CNOT}$ gate.

    Examples:
        >>> dq.cnot()
        QArray: shape=(4, 4), dims=(2, 2), dtype=complex64, layout=dense
        [[1.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 1.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 1.+0.j]
         [0.+0.j 0.+0.j 1.+0.j 0.+0.j]]
    """
    array = jnp.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=cdtype()
    )
    return asqarray(array, dims=(2, 2))


def toffoli() -> QArray:
    r"""Returns the $\text{Toffoli}$ gate.

    It is defined by
    $$
        \text{Toffoli} = \begin{pmatrix}
            1 & 0 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
            0 & 1 & 0 & 0 & 0 & 0 & 0 & 0 \\\\
            0 & 0 & 1 & 0 & 0 & 0 & 0 & 0 \\\\
            0 & 0 & 0 & 1 & 0 & 0 & 0 & 0 \\\\
            0 & 0 & 0 & 0 & 1 & 0 & 0 & 0 \\\\
            0 & 0 & 0 & 0 & 0 & 1 & 0 & 0 \\\\
            0 & 0 & 0 & 0 & 0 & 0 & 0 & 1 \\\\
            0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
        \end{pmatrix}
    $$

    Returns:
        _(qarray of shape (8, 8))_ $\text{Toffoli}$ gate.

    Examples:
        >>> dq.toffoli()
        QArray: shape=(8, 8), dims=(2, 2, 2), dtype=complex64, layout=dense
        [[1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j]
         [0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 0.+0.j 1.+0.j 0.+0.j]]
    """
    array = jnp.array(
        [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 0, 0, 1, 0],
        ],
        dtype=cdtype(),
    )
    return asqarray(array, dims=(2, 2, 2))
