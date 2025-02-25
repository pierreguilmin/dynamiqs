from __future__ import annotations

from functools import partial, reduce

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array

from .._checks import check_hermitian, check_shape
from ..qarrays.qarray import QArray, QArrayLike, get_dims
from ..qarrays.utils import asqarray, init_dims, to_jax

__all__ = [
    'bloch_coordinates',
    'braket',
    'cosm',
    'dag',
    'dissipator',
    'entropy_vn',
    'expect',
    'expm',
    'fidelity',
    'purity',
    'isbra',
    'isdm',
    'isherm',
    'isket',
    'isop',
    'lindbladian',
    'norm',
    'overlap',
    'powm',
    'proj',
    'ptrace',
    'sinm',
    'tensor',
    'tobra',
    'todm',
    'toket',
    'trace',
    'tracemm',
    'unit',
    'signm',
]


def dag(x: QArrayLike) -> QArray:
    r"""Returns the adjoint (complex conjugate transpose) of a matrix.

    Args:
        x _(qarray-like of shape (..., m, n))_: Matrix.

    Returns:
       _(qarray of shape (..., n, m))_ Adjoint of `x`.

    Note-: Equivalent syntax
        This function is equivalent to `x.mT.conj()`.

    Examples:
        >>> dq.fock(2, 0)
        QArray: shape=(2, 1), dims=(2,), dtype=complex64, layout=dense
        [[1.+0.j]
         [0.+0.j]]
        >>> dq.dag(dq.fock(2, 0))
        QArray: shape=(1, 2), dims=(2,), dtype=complex64, layout=dense
        [[1.-0.j 0.-0.j]]
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., m, n)')
    return x.mT.conj()


def powm(x: QArrayLike, n: int) -> QArray:
    """Returns the $n$-th matrix power of a qarray.

    Args:
        x _(qarray-like of shape (..., n, n))_: Square matrix.
        n: Integer exponent.

    Returns:
        _(qarray of shape (..., n, n))_ Matrix power of `x`.

    Examples:
        >>> dq.powm(dq.sigmax(), 2)
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dia, ndiags=1
        [[1.+0.j   ⋅   ]
         [  ⋅    1.+0.j]]
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, n)')
    return x.powm(n)


def expm(x: QArrayLike, *, max_squarings: int = 16) -> QArray:
    """Returns the matrix exponential of a qarray.

    The exponential is computed using the scaling-and-squaring approximation method.

    Args:
        x _(qarray-like of shape (..., n, n))_: Square matrix.
        max_squarings: Number of squarings.

    Returns:
        _(qarray of shape (..., n, n))_ Matrix exponential of `x`.

    Note-: Equivalent JAX syntax
        This function is equivalent to
        `jnp.scipy.linalg.expm(x, max_squarings=max_squarings)`.

    Examples:
        >>> dq.expm(dq.sigmaz())
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[2.718+0.j 0.   +0.j]
         [0.   +0.j 0.368+0.j]]
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, n)')
    return x.expm(max_squarings=max_squarings)


def cosm(x: QArrayLike) -> QArray:
    r"""Returns the cosine of a qarray.

    Args:
        x _(qarray-like of shape (..., n, n))_: Square matrix.

    Returns:
        _(qarray of shape (..., n, n))_ Cosine of `x`.

    Note:
        This function uses [`jax.scipy.linalg.expm()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.expm.html)
        to compute the cosine of a matrix $A$:
        $$
            \cos(A) = \frac{e^{iA} + e^{-iA}}{2}
        $$

    Examples:
        >>> dq.cosm(jnp.pi * dq.sigmax())
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[-1.+0.j  0.+0.j]
         [ 0.+0.j -1.+0.j]]
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, n)')
    return 0.5 * (expm(1j * x) + expm(-1j * x))


def sinm(x: QArrayLike) -> QArray:
    r"""Returns the sine of a qarray.

    Args:
        x _(qarray-like of shape (..., n, n))_: Square matrix.

    Returns:
        _(qarray of shape (..., n, n))_ Sine of `x`.

    Note:
        This function uses [`jax.scipy.linalg.expm()`](https://jax.readthedocs.io/en/latest/_autosummary/jax.scipy.linalg.expm.html)
        to compute the sine of a matrix $A$:
        $$
            \sin(A) = \frac{e^{iA} - e^{-iA}}{2i}
        $$

    Examples:
        >>> dq.sinm(0.5 * jnp.pi * dq.sigmax())
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[0.-0.j 1.-0.j]
         [1.-0.j 0.-0.j]]
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, n)')
    return -0.5j * (expm(1j * x) - expm(-1j * x))


def signm(x: QArrayLike) -> QArray:
    r"""Returns the operator sign function of a hermitian qarray.

    The operator sign function $\mathrm{sign}(A)$ of a hermitian matrix $A$ with
    eigendecomposition $A = U\, \text{diag}(\lambda_1,\dots,\lambda_n)\, U^\dagger$,
    with $(\lambda_1,\dots,\lambda_n)\in\R^n$ the eigenvalues of $A$, is defined by
    $$
        \mathrm{sign}(A) = U\,\mathrm{diag}(\mathrm{sign}(\lambda_1),\dots,\mathrm{sign}(\lambda_n))\,U^\dagger,
    $$
    where $\mathrm{sign}(x)$ is the sign of $x\in\R$.

    Args:
        x _(qarray-like of shape (..., n, n))_: Square hermitian matrix.

    Returns:
        _(qarray of shape (..., n, n))_ Operator sign function of `x`.

    Note:
        The operator sign is generally dense, and is different from the element-wise
        sign of the operator.

    Examples:
        >>> dq.signm(dq.sigmax())
        QArray: shape=(2, 2), dims=(2,), dtype=complex64, layout=dense
        [[0.+0.j 1.+0.j]
         [1.+0.j 0.+0.j]]
        >>> dq.position(3)
        QArray: shape=(3, 3), dims=(3,), dtype=complex64, layout=dia, ndiags=2
        [[    ⋅     0.5  +0.j     ⋅    ]
         [0.5  +0.j     ⋅     0.707+0.j]
         [    ⋅     0.707+0.j     ⋅    ]]
        >>> dq.signm(dq.position(3))
        QArray: shape=(3, 3), dims=(3,), dtype=complex64, layout=dense
        [[-0.667+0.j  0.577+0.j  0.471+0.j]
         [ 0.577+0.j  0.   +0.j  0.816+0.j]
         [ 0.471+0.j  0.816+0.j -0.333+0.j]]
    """  # noqa: E501
    x = asqarray(x)
    x = check_hermitian(x, 'x')
    L, Q = x.asdense()._eigh()
    sign_L = jnp.diag(jnp.sign(L))
    array = Q @ sign_L @ dag(Q)
    return asqarray(array, dims=x.dims)


def trace(x: QArrayLike) -> Array:
    r"""Returns the trace of a qarray along its last two dimensions.

    Args:
        x _(qarray-like of shape (..., n, n))_: Qarray-like.

    Returns:
        _(array of shape (...))_ Trace of `x`.

    Examples:
        >>> x = jnp.ones((3, 3))
        >>> dq.trace(x)
        Array(3., dtype=float32)
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, n)')
    return x.trace()


def tracemm(x: QArrayLike, y: QArrayLike) -> Array:
    r"""Return the trace of a matrix multiplication using a fast implementation.

    The trace is computed as `sum(x * y.T)` where `*` is the element-wise product,
    instead of `trace(x @ y)` where `@` is the matrix product. Indeed, we have:

    $$
        \tr{xy} = \sum_i (xy)_{ii}
                = \sum_{i,j} x_{ij} y_{ji}
                = \sum_{i,j} x_{ij} (y^\intercal)_{ij}
                = \sum_{i,j} (x * y^\intercal)_{ij}
    $$

    Note:
        The resulting time complexity for $n\times n$ matrices is $\mathcal{O}(n^2)$
        instead of $\mathcal{O}(n^3)$ with the naive formula.

    Args:
        x _(qarray-like of shape (..., n, n))_: Qarray-like.
        y _(qarray-like of shape (..., n, n))_: Qarray-like.

    Returns:
        _(array of shape (...))_ Trace of `x @ y`.

    Examples:
        >>> x = jnp.ones((3, 3))
        >>> y = jnp.ones((3, 3))
        >>> dq.tracemm(x, y)
        Array(9., dtype=float32)
    """
    x = asqarray(x)
    y = asqarray(y)
    check_shape(x, 'x', '(..., n, n)')
    check_shape(y, 'y', '(..., n, n)')
    # todo: fix perf
    return (x.to_jax() * y.to_jax().mT).sum((-2, -1))


def _hdim(x: QArrayLike) -> int:
    x = asqarray(x)
    return x.shape[-2] if isket(x) else x.shape[-1]


@partial(jax.jit, static_argnums=(1, 2))
def ptrace(
    x: QArrayLike, keep: int | tuple[int, ...], dims: tuple[int, ...] | None = None
) -> QArray:
    r"""Returns the partial trace of a ket, bra or density matrix.

    Args:
        x _(qarray-like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra
            or density matrix of a composite system.
        keep _(int or tuple of ints)_: Dimensions to keep after partial trace.
        dims _(tuple of ints or None)_: Dimensions of each subsystem in the composite
            system Hilbert space tensor product. Defaults to `None` (`x.dims` if
            available, single Hilbert space `dims=(n,)` otherwise).

    Returns:
        _(qarray of shape (..., m, m))_ Density matrix (with `m <= n`).

    Note:
        The returned object is always a density matrix, even if the input is a ket or a
        bra.

    Examples:
        >>> psi_abc = dq.fock((3, 4, 5), (0, 2, 1))
        >>> psi_abc.dims
        (3, 4, 5)
        >>> psi_abc.shape
        (60, 1)
        >>> rho_a = dq.ptrace(psi_abc, 0)
        >>> rho_a.dims
        (3,)
        >>> rho_a.shape
        (3, 3)
        >>> rho_bc = dq.ptrace(psi_abc, (1, 2))
        >>> rho_bc.dims
        (4, 5)
        >>> rho_bc.shape
        (20, 20)

        If the input qarray-like `x` does not hold Hilbert space dimensions, you
        can specify them with the argument `dims`. For example, to trace out the second
        subsystem of the Bell state $(\ket{00}+\ket{11})/\sqrt2$:
        >>> bell_state = np.array([1, 0, 0, 1])[:, None] / np.sqrt(2)
        >>> bell_state.shape
        (4, 1)
        >>> dq.ptrace(bell_state, 0, dims=(2, 2))
        QArray: shape=(2, 2), dims=(2,), dtype=float32, layout=dense
        [[0.5 0. ]
         [0.  0.5]]
    """
    xdims = get_dims(x)
    x = to_jax(x)
    dims = init_dims(xdims, dims, x.shape)
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)', '(..., n, n)')

    # convert keep and dims to numpy arrays
    keep = np.asarray([keep] if isinstance(keep, int) else keep)  # e.g. [1, 2]
    dims = np.asarray(dims)  # e.g. [20, 2, 5]
    ndims = len(dims)  # e.g. 3

    # check that input dimensions match
    hdim = _hdim(x)
    prod_dims = np.prod(dims)
    if prod_dims != hdim:
        dims_prod_str = '*'.join(str(d) for d in dims) + f'={prod_dims}'
        raise ValueError(
            'Argument `dims` must match the Hilbert space dimension of `x` of'
            f' {hdim}, but the product of its values is {dims_prod_str}.'
        )
    if np.any(keep < 0) or np.any(keep > len(dims) - 1):
        raise ValueError(
            'Argument `keep` must match the Hilbert space structure specified by'
            ' `dims`.'
        )

    # sort keep
    keep.sort()

    # create einsum alphabet
    alphabet = 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ'

    # compute einsum equations
    eq1 = alphabet[:ndims]  # e.g. 'abc'
    unused = iter(alphabet[ndims:])
    eq2 = ''.join(
        [next(unused) if i in keep else eq1[i] for i in range(ndims)]
    )  # e.g. 'ade'

    bshape = x.shape[:-2]

    # trace out x over unkept dimensions
    if isket(x) or isbra(x):
        x = x.reshape(*bshape, *dims)  # e.g. (..., 20, 2, 5)
        eq = f'...{eq1},...{eq2}'  # e.g. '...abc,...ade'
        x = jnp.einsum(eq, x, x.conj())  # e.g. (..., 2, 5, 2, 5)
    else:
        x = x.reshape(*bshape, *dims, *dims)  # e.g. (..., 20, 2, 5, 20, 2, 5)
        eq = f'...{eq1}{eq2}'  # e.g. '...abcade'
        x = jnp.einsum(eq, x)  # e.g. (..., 2, 5, 2, 5)

    new_dims = tuple(dims[keep].tolist())
    prod_new_dims = np.prod(new_dims)  # e.g. 10
    x = x.reshape(*bshape, prod_new_dims, prod_new_dims)  # e.g. (..., 10, 10)

    return asqarray(x, dims=new_dims)


def tensor(*args: QArrayLike) -> QArray:
    r"""Returns the tensor product of multiple kets, bras, density matrices or
    operators.

    The returned qarray shape is:

    - $(..., n, 1)$ with $n=\prod_k n_k$ if all input qarrays are kets with shape
      $(..., n_k, 1)$,
    - $(..., 1, n)$ with $n=\prod_k n_k$ if all input qarrays are bras with shape
      $(..., 1, n_k)$,
    - $(..., n, n)$ with $n=\prod_k n_k$ if all input qarrays are density matrices or
      operators with shape $(..., n_k, n_k)$.

    Args:
        *args _(qarray-like of shape (..., n_k, 1) or (..., 1, n_k) or (..., n_k, n_k))_:
            Variable length argument list of kets, bras, density matrices or operators.

    Returns:
        _(qarray of shape (..., n, 1) or (..., 1, n) or (..., n, n))_ Tensor product of
            the input qarrays.

    Examples:
        >>> psi = dq.tensor(dq.fock(3, 0), dq.fock(4, 2), dq.fock(5, 1))
        >>> psi.shape
        (60, 1)
    """  # noqa: E501
    args = [asqarray(arg) for arg in args]
    return reduce(lambda x, y: x & y, args)  # TODO: (guilmin) use jax.lax.reduce


def expect(O: QArrayLike, x: QArrayLike) -> Array:
    r"""Returns the expectation value of an operator or list of operators on a ket, bra
    or density matrix.

    The expectation value $\braket{O}$ of an operator $O$ is computed

    - as $\braket{O}=\braket{\psi|O|\psi}$ if `x` is a ket $\ket\psi$ or bra $\bra\psi$,
    - as $\braket{O}=\tr{O\rho}$ if `x` is a density matrix $\rho$.

    Warning:
        The returned array is complex-valued. If the operator $O$ corresponds to a
        physical observable, it is Hermitian: $O^\dag=O$, and the expectation value
        is real. One can then keep only the real values of the returned array using
        `dq.expect(O, x).real`.

    Args:
        O _(qarray-like of shape (nO?, n, n))_: Arbitrary operator or list of _nO_
            operators.
        x _(qarray-like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket,
            bra or density matrix.

    Returns:
        _(array of shape (nO?, ...))_ Complex-valued expectation value.

    Examples:
        >>> O = dq.number(16)
        >>> psi = dq.coherent(16, 2.0)
        >>> dq.expect(O, psi)
        Array(4.+0.j, dtype=complex64)
        >>> psis = [dq.fock(16, i) for i in range(5)]
        >>> dq.expect(O, psis).shape
        (5,)
        >>> Os = [dq.position(16), dq.momentum(16)]
        >>> dq.expect(Os, psis).shape
        (2, 5)
    """
    O = asqarray(O)
    x = asqarray(x)
    check_shape(O, 'O', '(?, n, n)', subs={'?': 'nO?'})
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)', '(..., n, n)')

    f = _expect_single
    if O.ndim > 2:
        f = jax.vmap(f, in_axes=(0, None))
    return f(O, x)


def _expect_single(O: QArray, x: QArray) -> Array:
    # O: (n, n), x: (..., n, m)
    if isket(x):
        return (dag(x) @ O @ x).squeeze((-1, -2))  # <x|O|x>
    elif isbra(x):
        return (x @ O @ dag(x)).squeeze((-1, -2))
    else:
        return tracemm(O, x)  # tr(Ox)


def norm(x: QArrayLike) -> Array:
    r"""Returns the norm of a ket, bra or density matrix.

    For a ket or a bra, the returned norm is $\sqrt{\braket{\psi|\psi}}$. For a density
    matrix, it is $\tr{\rho}$.

    Args:
        x _(qarray-like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra
            or density matrix.

    Returns:
        _(array of shape (...))_ Real-valued norm of `x`.

    Examples:
        For a ket:
        >>> psi = dq.fock(4, 0) + dq.fock(4, 1)
        >>> dq.norm(psi)
        Array(1.414, dtype=float32)

        For a density matrix:
        >>> rho = dq.fock_dm(4, 0) + dq.fock_dm(4, 1) + dq.fock_dm(4, 2)
        >>> dq.norm(rho)
        Array(3., dtype=float32)
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)', '(..., n, n)')

    if isket(x) or isbra(x):
        return jnp.sqrt((jnp.abs(x.to_jax()) ** 2).sum((-2, -1)))
    else:
        return trace(x).real


def unit(x: QArrayLike) -> QArray:
    r"""Normalize a ket, bra or density matrix to unit norm.

    The returned object is divided by its norm (see [`dq.norm()`][dynamiqs.norm]).

    Args:
        x _(qarray-like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra
            or density matrix.

    Returns:
        _(qarray of shape (..., n, 1) or (..., 1, n) or (..., n, n))_ Normalized ket,
            bra or density matrix.

    Examples:
        >>> psi = dq.fock(4, 0) + dq.fock(4, 1)
        >>> psi.norm()
        Array(1.414, dtype=float32)
        >>> psi = dq.unit(psi)
        >>> psi.norm()
        Array(1., dtype=float32)
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)', '(..., n, n)')
    return x / norm(x)[..., None, None]


def dissipator(L: QArrayLike, rho: QArrayLike) -> QArray:
    r"""Applies the Lindblad dissipation superoperator to a density matrix.

    The dissipation superoperator $\mathcal{D}[L]$ is defined by:
    $$
        \mathcal{D}[L] (\rho) = L\rho L^\dag - \frac{1}{2}L^\dag L \rho
        - \frac{1}{2}\rho L^\dag L.
    $$

    Args:
        L _(qarray-like of shape (..., n, n))_: Jump operator.
        rho _(qarray-like of shape (..., n, n))_: Density matrix.

    Returns:
        _(qarray of shape (..., n, n))_ Resulting operator (it is not a density matrix).

    See also:
        - [`dq.sdissipator()`][dynamiqs.sdissipator]: returns the dissipation
            superoperator in matrix form (vectorized).

    Examples:
        >>> L = dq.destroy(4)
        >>> rho = dq.fock_dm(4, 2)
        >>> dq.dissipator(L, rho)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dense
        [[ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  2.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j -2.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]]
    """
    L = asqarray(L)
    rho = asqarray(rho)
    check_shape(L, 'L', '(..., n, n)')
    check_shape(rho, 'rho', '(..., n, n)')

    Ldag = dag(L)
    LdagL = Ldag @ L
    return L @ rho @ Ldag - 0.5 * LdagL @ rho - 0.5 * rho @ LdagL


def lindbladian(H: QArrayLike, jump_ops: list[QArrayLike], rho: QArrayLike) -> QArray:
    r"""Applies the Lindbladian superoperator to a density matrix.

    The Lindbladian superoperator $\mathcal{L}$ is defined by:
    $$
        \mathcal{L} (\rho) = -i[H,\rho] + \sum_{k=1}^N \mathcal{D}[L_k] (\rho),
    $$

    where $H$ is the system Hamiltonian, $\{L_k\}$ is a set of $N$ jump operators
    (arbitrary operators) and $\mathcal{D}[L]$ is the Lindblad dissipation superoperator
    (see [`dq.dissipator()`][dynamiqs.dissipator]).

    Note:
        This superoperator is also sometimes called *Liouvillian*.

    Args:
        H _(qarray-like of shape (..., n, n))_: Hamiltonian.
        jump_ops _(list of qarray-like, each of shape (, ..., n, n))_: List of jump
            operators.
        rho _(qarray-like of shape (..., n, n))_: Density matrix.

    Returns:
        _(qarray of shape (..., n, n))_ Resulting operator (it is not a density matrix).

    See also:
        - [`dq.slindbladian()`][dynamiqs.slindbladian]: returns the Lindbladian
            superoperator in matrix form (vectorized).

    Examples:
        >>> a = dq.destroy(4)
        >>> H = a.dag() @ a
        >>> L = [a, a.dag() @ a]
        >>> rho = dq.fock_dm(4, 1)
        >>> dq.lindbladian(H, L, rho)
        QArray: shape=(4, 4), dims=(4,), dtype=complex64, layout=dense
        [[ 1.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j -1.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]
         [ 0.+0.j  0.+0.j  0.+0.j  0.+0.j]]
    """
    H = asqarray(H)
    jump_ops = [asqarray(L) for L in jump_ops]
    rho = asqarray(rho)

    # === check H shape
    check_shape(H, 'H', '(..., n, n)')

    # === check jump_ops shape
    for i, L in enumerate(jump_ops):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)')

    # === check rho shape
    check_shape(rho, 'rho', '(..., n, n)')

    return -1j * H @ rho + 1j * rho @ H + sum([dissipator(L, rho) for L in jump_ops])


def isket(x: QArrayLike) -> bool:
    r"""Returns True if the qarray is in the format of a ket.

    Args:
        x _(qarray-like of shape (...))_: Qarray-like.

    Returns:
        True if the last dimension of `x` is 1, False otherwise.

    Examples:
        >>> dq.isket(jnp.ones((3, 1)))
        True
        >>> dq.isket(jnp.ones((5, 3, 1)))
        True
        >>> dq.isket(jnp.ones((3, 3)))
        False
    """
    x = asqarray(x)
    return x.shape[-1] == 1


def isbra(x: QArrayLike) -> bool:
    r"""Returns True if the qarray is in the format of a bra.

    Args:
        x _(qarray-like of shape (...))_: Qarray-like.

    Returns:
        True if the second to last dimension of `x` is 1, False otherwise.

    Examples:
        >>> dq.isbra(jnp.ones((1, 3)))
        True
        >>> dq.isbra(jnp.ones((5, 1, 3)))
        True
        >>> dq.isbra(jnp.ones((3, 3)))
        False
    """
    x = asqarray(x)
    return x.shape[-2] == 1


def isdm(x: QArrayLike) -> bool:
    r"""Returns True if the qarray is in the format of a density matrix.

    Args:
        x _(qarray-like of shape (...))_: Qarray-like.

    Returns:
        True if the last two dimensions of `x` are equal, False otherwise.

    Examples:
        >>> dq.isdm(jnp.ones((3, 3)))
        True
        >>> dq.isdm(jnp.ones((5, 3, 3)))
        True
        >>> dq.isdm(jnp.ones((3, 1)))
        False
    """
    x = asqarray(x)
    return x.shape[-1] == x.shape[-2]


def isop(x: QArrayLike) -> bool:
    r"""Returns True if the qarray is in the format of an operator.

    Args:
        x _(qarray-like of shape (...))_: Qarray-like.

    Returns:
        True if the last two dimensions of `x` are equal, False otherwise.

    Examples:
        >>> dq.isop(jnp.ones((3, 3)))
        True
        >>> dq.isop(jnp.ones((5, 3, 3)))
        True
        >>> dq.isop(jnp.ones((3, 1)))
        False
    """
    x = asqarray(x)
    return x.shape[-1] == x.shape[-2]


def isherm(x: QArrayLike, rtol: float = 1e-5, atol: float = 1e-8) -> bool:
    r"""Returns True if the qarray is Hermitian.

    Args:
        x _(qarray-like of shape (..., n, n))_: Qarray-like.
        rtol: Relative tolerance of the check.
        atol: Absolute tolerance of the check.

    Returns:
        True if `x` is Hermitian, False otherwise.

    Examples:
        >>> dq.isherm(jnp.eye(3))
        Array(True, dtype=bool)
        >>> dq.isherm([[0, 1j], [1j, 0]])
        Array(False, dtype=bool)
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, n)')
    return x.isherm(rtol=rtol, atol=atol)


def toket(x: QArrayLike) -> QArray:
    r"""Returns the ket representation of a pure quantum state.

    Args:
        x _(qarray-like of shape (..., n, 1) or (..., 1, n))_: Ket or bra.

    Returns:
        _(qarray of shape (..., n, 1))_ Ket.

    Examples:
        >>> psi = dq.fock(3, 0).tobra()  # shape: (1, 3)
        >>> psi
        QArray: shape=(1, 3), dims=(3,), dtype=complex64, layout=dense
        [[1.-0.j 0.-0.j 0.-0.j]]
        >>> dq.toket(psi)  # shape: (3, 1)
        QArray: shape=(3, 1), dims=(3,), dtype=complex64, layout=dense
        [[1.+0.j]
         [0.+0.j]
         [0.+0.j]]
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)')

    if isbra(x):
        return dag(x)
    else:
        return x


def tobra(x: QArrayLike) -> QArray:
    r"""Returns the bra representation of a pure quantum state.

    Args:
        x _(qarray-like of shape (..., n, 1) or (..., 1, n))_: Ket or bra.

    Returns:
        _(qarray of shape (..., 1, n))_ Qarray.

    Examples:
        >>> psi = dq.fock(3, 0)  # shape: (3, 1)
        >>> psi
        QArray: shape=(3, 1), dims=(3,), dtype=complex64, layout=dense
        [[1.+0.j]
         [0.+0.j]
         [0.+0.j]]
        >>> dq.tobra(psi)  # shape: (1, 3)
        QArray: shape=(1, 3), dims=(3,), dtype=complex64, layout=dense
        [[1.-0.j 0.-0.j 0.-0.j]]
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)')

    if isbra(x):
        return x
    else:
        return dag(x)


def todm(x: QArrayLike) -> QArray:
    r"""Returns the density matrix representation of a quantum state.

    Note:
        This function is an alias of [`dq.proj()`][dynamiqs.proj]. If `x` is already a
        density matrix, it is returned directly.

    Args:
        x _(qarray-like of shape (..., n, 1) or (..., 1, n) or (..., n, n))_: Ket, bra
            or density matrix.

    Returns:
        _(qarray of shape (..., n, n))_ Density matrix.

    Examples:
        >>> psi = dq.fock(3, 0)  # shape: (3, 1)
        >>> psi
        QArray: shape=(3, 1), dims=(3,), dtype=complex64, layout=dense
        [[1.+0.j]
         [0.+0.j]
         [0.+0.j]]
        >>> dq.todm(psi)  # shape: (3, 3)
        QArray: shape=(3, 3), dims=(3,), dtype=complex64, layout=dense
        [[1.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j]]
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)', '(..., n, n)')

    if isbra(x) or isket(x):
        return proj(x)
    else:
        return x


def proj(x: QArrayLike) -> QArray:
    r"""Returns the projection operator onto a pure quantum state.

    The projection operator onto the state $\ket\psi$ is defined as
    $P_{\ket\psi} = \ket\psi\bra\psi$.

    Args:
        x _(qarray-like of shape (..., n, 1) or (..., 1, n))_: Ket or bra.

    Returns:
        _(qarray of shape (..., n, n))_ Projection operator.

    Examples:
        >>> psi = dq.fock(3, 0)
        >>> dq.proj(psi)
        QArray: shape=(3, 3), dims=(3,), dtype=complex64, layout=dense
        [[1.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j]
         [0.+0.j 0.+0.j 0.+0.j]]
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, 1)', '(..., 1, n)')

    if isbra(x):
        return dag(x) @ x
    else:
        return x @ dag(x)


def braket(x: QArrayLike, y: QArrayLike) -> Array:
    r"""Returns the inner product $\braket{\psi|\varphi}$ between two kets.

    Args:
        x (qarray-like of shape _(..., n, 1))_: Left-side ket.
        y (qarray-like of shape _(..., n, 1))_: Right-side ket.

    Returns:
        _(array of shape (...))_ Complex-valued inner product.

    Examples:
        >>> fock0 = dq.fock(3, 0)
        >>> fock01 = (dq.fock(3, 0) + dq.fock(3, 1)).unit()
        >>> dq.braket(fock0, fock01)
        Array(0.707+0.j, dtype=complex64)
    """
    x = asqarray(x)
    y = asqarray(y)
    check_shape(x, 'x', '(..., n, 1)')
    check_shape(y, 'y', '(..., n, 1)')

    return (dag(x) @ y).squeeze((-1, -2))


def overlap(x: QArrayLike, y: QArrayLike) -> Array:
    r"""Returns the overlap between two quantum states.

    The overlap is computed

    - as $\lvert\braket{\psi|\varphi}\rvert^2$ if both arguments are kets $\ket\psi$
      and $\ket\varphi$,
    - as $\lvert\bra\psi \rho \ket\psi\rvert$ if one argument is a ket $\ket\psi$ and
      the other is a density matrix $\rho$,
    - as $\tr{\rho^\dag\sigma}$ if both arguments are density matrices $\rho$ and
      $\sigma$.

    Args:
        x _(qarray-like of shape (..., n, 1) or (..., n, n))_: Ket or density matrix.
        y _(qarray-like of shape (..., n, 1) or (..., n, n))_: Ket or density matrix.

    Returns:
        _(array of shape (...))_ Real-valued overlap.

    Examples:
        >>> fock0 = dq.fock(3, 0)
        >>> dq.overlap(fock0, fock0)
        Array(1., dtype=float32)
        >>> fock01_dm = 0.5 * (dq.fock_dm(3, 0) + dq.fock_dm(3, 1))
        >>> dq.overlap(fock0, fock01_dm)
        Array(0.5, dtype=float32)
    """
    x = asqarray(x)
    y = asqarray(y)
    check_shape(x, 'x', '(..., n, 1)', '(..., n, n)')
    check_shape(y, 'y', '(..., n, 1)', '(..., n, n)')

    if isket(x) and isket(y):
        return jnp.abs((dag(x) @ y).squeeze((-1, -2))) ** 2
    elif isket(x):
        return jnp.abs((dag(x) @ y @ x).squeeze((-1, -2)))
    elif isket(y):
        return jnp.abs((dag(y) @ x @ y).squeeze((-1, -2)))
    else:
        return tracemm(dag(x), y).real


def fidelity(x: QArrayLike, y: QArrayLike) -> Array:
    r"""Returns the fidelity of two states, kets or density matrices.

    The fidelity is computed

    - as $F(\ket\psi,\ket\varphi)=\left|\braket{\psi|\varphi}\right|^2$ if both
      arguments are kets,
    - as $F(\ket\psi,\rho)=\lvert\braket{\psi|\rho|\psi}\rvert$ if one arguments is a
      ket and the other is a density matrix,
    - as $F(\rho,\sigma)=\tr{\sqrt{\sqrt\rho\sigma\sqrt\rho}}^2$ if both arguments are
      density matrices.

    Warning:
        This definition is different from `qutip.fidelity()` which uses the square root
        fidelity $F_\text{qutip} = \sqrt{F}$.

    Args:
        x _(qarray-like of shape (..., n, 1) or (..., n, n))_: Ket or density matrix.
        y _(qarray-like of shape (..., n, 1) or (..., n, n))_: Ket or density matrix.

    Returns:
        _(array of shape (...))_ Real-valued fidelity.

    Examples:
        >>> fock0 = dq.fock(3, 0)
        >>> dq.fidelity(fock0, fock0)
        Array(1., dtype=float32)
        >>> fock01_dm = 0.5 * (dq.fock_dm(3, 1) + dq.fock_dm(3, 0))
        >>> dq.fidelity(fock01_dm, fock01_dm)
        Array(1., dtype=float32)
        >>> dq.fidelity(fock0, fock01_dm)
        Array(0.5, dtype=float32)
    """
    x = asqarray(x)
    y = asqarray(y)
    check_shape(x, 'x', '(..., n, 1)', '(..., n, n)')
    check_shape(y, 'y', '(..., n, 1)', '(..., n, n)')

    if isket(x) or isket(y):
        return overlap(x, y)
    else:
        return _dm_fidelity(x, y)


def _dm_fidelity(x: QArray, y: QArray) -> Array:
    # returns the fidelity of two density matrices: Tr[sqrt(sqrt(x) @ y @ sqrt(x))]^2
    # x: (..., n, n), y: (..., n, n) -> (...)

    # Note that the fidelity can be rewritten as Tr[sqrt(x @ y)]^2 (see
    # https://arxiv.org/abs/2211.02623). This is cheaper numerically, we just compute
    # the eigenvalues w_i of the matrix product x @ y and compute the fidelity as
    # F = (\sum_i \sqrt{w_i})^2.

    # note that we can't use `eigvalsh` here because x @ y is not necessarily Hermitian
    w = (x @ y)._eigvals().real
    # we set small negative eigenvalues errors to zero to avoid `nan` propagation
    w = jnp.where(w < 0, 0, w)
    return jnp.sqrt(w).sum(-1) ** 2


def purity(x: QArrayLike) -> Array:
    r"""Returns the purity of a ket or density matrix.

    For a ket (a pure state), the purity is $1$. For a density matrix $\rho$, it is
    defined by $\tr{\rho^2}$.

    Args:
        x _(qarray-like of shape (..., n, 1) or (..., n, n))_: Ket or density matrix.

    Returns:
        _(array of shape (...))_ Real-valued purity.

    Examples:
        >>> psi = dq.fock(2, 0)
        >>> dq.purity(psi)
        Array(1., dtype=float32)
        >>> rho = (dq.fock_dm(2, 0) + dq.fock_dm(2, 1)).unit()
        >>> dq.purity(rho)
        Array(0.5, dtype=float32)
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, 1)', '(..., n, n)')
    if x.isket():
        return jnp.ones(x.shape[:-2])
    return tracemm(x, x).real


def entropy_vn(x: QArrayLike) -> Array:
    r"""Returns the Von Neumann entropy of a ket or density matrix.

    It is defined by $S(\rho) = -\tr{\rho \ln \rho}$.

    Args:
        x _(qarray-like of shape (..., n, 1) or (..., n, n))_: Ket or density matrix.

    Returns:
        _(array of shape (...))_ Real-valued Von Neumann entropy.

    Examples:
        >>> rho = (dq.fock_dm(2, 0) + dq.fock_dm(2, 1)).unit()
        >>> dq.entropy_vn(rho)
        Array(0.693, dtype=float32)
        >>> psis = [dq.fock(16, i) for i in range(5)]
        >>> dq.entropy_vn(psis).shape
        (5,)
    """
    x = asqarray(x)
    check_shape(x, 'x', '(..., n, 1)', '(..., n, n)')

    if isket(x):
        return jnp.zeros(x.shape[:-2])

    # compute sum(w_i log(w_i)) where w_i are rho's eigenvalues
    w = x._eigvalsh()
    # we set small negative or null eigenvalues to 1.0 to avoid `nan` propagation
    w = jnp.where(w <= 0, 1.0, w)
    return -(w * jnp.log(w)).sum(-1)


def bloch_coordinates(x: QArrayLike) -> Array:
    r"""Returns the spherical coordinates $(r, \theta, \phi)$ of a ket or density matrix
    on the Bloch sphere.

    The coordinates are such that
    $$
        \begin{cases}
            0\leq r \leq 1, \\\\
            0\leq\theta\leq\pi, \\\\
            0\leq\phi<2\pi.
        \end{cases}
    $$

    By convention, we choose $\phi=0$ if $\theta=0$, and $\theta=\phi=0$ if $r=0$.

    Args:
        x _(qarray-like of shape (2, 1) or (2, 2))_: Ket or density matrix.

    Returns:
        _(array of shape (3,))_ Spherical coordinates $(r, \theta, \phi)$.

    Examples:
        The state $\ket0$ is on the north pole at coordinates
        $(r,\theta,\phi) = (1,0,0)$:
        >>> x = dq.basis(2, 0)
        >>> dq.bloch_coordinates(x)
        Array([1., 0., 0.], dtype=float32)

        The state $\ket1$ is on the south pole at coordinates
        $(r,\theta,\phi) = (1,\pi,0)$:
        >>> x = dq.basis(2, 1)
        >>> dq.bloch_coordinates(x)
        Array([1.   , 3.142, 0.   ], dtype=float32)

        The state $\ket+=(\ket0+\ket1)/\sqrt2$ is aligned with the $x$-axis at
        coordinates $(r,\theta,\phi) = (1,\pi/2,0)$:
        >>> plus = dq.unit(dq.basis(2, 0) + dq.basis(2, 1))
        >>> dq.bloch_coordinates(plus)
        Array([1.   , 1.571, 0.   ], dtype=float32)

        The state $\ket-=(\ket0-\ket1)/\sqrt2$ is aligned with the $-x$-axis at
        coordinates $(r,\theta,\phi) = (1,\pi/2,\pi)$:
        >>> minus = dq.unit(dq.basis(2, 0) - dq.basis(2, 1))
        >>> dq.bloch_coordinates(minus)
        Array([1.   , 1.571, 3.142], dtype=float32)

        A fully mixed state $\rho=0.5\ket0\bra0+0.5\ket1\bra1$ is at the center of the
        sphere at coordinates $(r,\theta,\phi) = (0,0,0)$:
        >>> x = 0.5 * dq.basis_dm(2, 0) + 0.5 * dq.basis_dm(2, 1)
        >>> dq.bloch_coordinates(x)
        Array([0., 0., 0.], dtype=float32)

        A partially mixed state $\rho=0.75\ket0\bra0 + 0.25\ket1\bra1$ is halfway
        between the sphere center and the north pole at coordinates
        $(r,\theta,\phi) = (0.5,0,0)$:
        >>> x = 0.75 * dq.basis_dm(2, 0) + 0.25 * dq.basis_dm(2, 1)
        >>> dq.bloch_coordinates(x)
        Array([0.5, 0. , 0. ], dtype=float32)
    """
    x = to_jax(x)  # todo: temporary fix
    check_shape(x, 'x', '(2, 1)', '(2, 2)')

    if isket(x):
        # Quick derivation: x = a |0> + b |1> with a = ra e^{i ta} and b = rb e^{i tb},
        # where ra^2 + rb^2 = 1. The state remains unchanged by a phase factor, so
        # e^{-i ta} x = ra |0> + rb e^{i (tb - ta)} |1>
        #             = cos(theta/2) |0> + e^{i phi} sin(theta/2) |1>
        # with theta = 2 * acos(ra) and phi = tb - ta (if rb != 0 else 0)
        a, b = x[:, 0]
        ra, ta = jnp.abs(a), jnp.angle(a)
        rb, tb = jnp.abs(b), jnp.angle(b)
        r = 1  # for a pure state
        theta = 2 * jnp.acos(ra)
        phi = tb - ta if rb != 0 else 0.0
    elif isdm(x):
        # cartesian coordinates
        # see https://en.wikipedia.org/wiki/Bloch_sphere#u,_v,_w_representation
        rx = 2 * x[0, 1].real
        ry = 2 * x[1, 0].imag
        rz = (x[0, 0] - x[1, 1]).real

        # spherical coordinates
        r = jnp.linalg.norm(jnp.array([rx, ry, rz]))
        if r == 0:
            theta = 0.0
            phi = 0.0
        else:
            theta = jnp.acos(rz / r)
            phi = jnp.arctan2(ry, rx)

    # map phi to [0, 2pi[
    phi = phi % (2 * jnp.pi)

    return jnp.array([r, theta, phi])
