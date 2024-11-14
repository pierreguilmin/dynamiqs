from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike, ScalarLike

from ..._checks import check_shape, check_times
from ...gradient import Gradient
from ...integrators.floquet.floquet_integrator import FloquetIntegrator
from ...options import Options
from ...result import FloquetResult
from ...solver import Solver, Tsit5
from ...time_array import TimeArray
from .._utils import _astimearray, cartesian_vmap, catch_xla_runtime_error

__all__ = ['floquet']


def floquet(
    H: ArrayLike | TimeArray,
    T: ScalarLike,
    tsave: ArrayLike,
    *,
    solver: Solver = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> FloquetResult:
    r"""Compute Floquet modes $\Phi_{m}(t)$ and quasienergies $\epsilon_m$.

    For a periodically driven system, the Floquet modes $\Phi_{m}(t_0)$ and
    quasienergies $\epsilon_m$ are defined by the eigenvalue equation
    $$
        U(t_0, t_0+T)\Phi_{m}(t_0) = \exp(-i \epsilon_{m} T)\Phi_{m}(t_0),
    $$
    where $U(t_0, t_0+T)$ is the propagator from time $t_0$ to time $t_0+T$, and $T$ is
    the period of the drive. Typically $t_0$ is taken to be $0$, however that does not
    not always have to be the case. We thus obtain the modes $\Phi_{m}(t_0)$ and
    quasienergies $\epsilon_m$ by diagonalizing the propagator $U(t_0, t_0+T)$.

    The Floquet modes $\Phi_{m}(t)$ at times $t\neq t_0$ are obtained from the Floquet
    modes $\Phi_{m}(t_0)$ via
    $$
        \Phi_{m}(t) = \exp(i\epsilon_{m}t)U(t_0, t_0+t)\Phi_{m}(t_0).
    $$

    Warning:
        The Floquet modes are only defined up to integer multiples of the drive period.
        One could envision then taking `tsave = jnp.mod(tsave, T)` in case times are
        provided that are greater than a single drive period, cutting down on
        integration time. We cannot do this here because `tsave` is passed to
        `sepropagator`, which expects the times in `tsave` to be ordered in strictly
        ascending order.

    Args:
        H _(array-like or time-array of shape (...H, n, n))_: Hamiltonian.
        T _(array-like of shape (...H))_: Period of the drive. T should have the same
            shape as ...H or should be broadcastable to that shape. This is to allow
            batching over Hamiltonians with differing drive frequencies.
        tsave _(array-like of shape (ntsave,)_: Times at which to compute floquet modes.
        solver: Solver for the integration.
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`][dynamiqs.Options].

    Returns:
        [`dq.FloquetResult`][dynamiqs.FloquetResult] object holding the result of the
            Floquet computation. Use the attribute `modes` to access the saved
            Floquet modes, and the attribute `quasienergies` the associated quasi
            energies, more details in [`dq.FloquetResult`][dynamiqs.FloquetResult].
    """
    # === convert arguments
    H = _astimearray(H)
    T = jnp.asarray(T)
    tsave = jnp.asarray(tsave)

    # === broadcast arguments
    # Different batch Hamiltonians may be periodic with varying periods, so T must be
    # broadcastable to the same shape as H.
    broadcast_shape = jnp.broadcast_shapes(H.shape[:-2], T.shape)
    H = H.broadcast_to(*(broadcast_shape + H.shape[-2:]))
    T = jnp.broadcast_to(T, broadcast_shape)

    # === check arguments
    tsave = check_times(tsave, 'tsave')
    H, T, tsave = _check_floquet_args(H, T, tsave)

    # We implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to JAX arrays
    return _vectorized_floquet(H, T, tsave, solver, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('solver', 'gradient', 'options'))
def _vectorized_floquet(
    H: TimeArray,
    T: Array,
    tsave: Array,
    solver: Solver,
    gradient: Gradient,
    options: Options,
) -> FloquetResult:
    # vectorize input over H
    in_axes = (H.in_axes, None, None, None, None, None)
    # vectorize output over `_saved` and `infos`
    out_axes = FloquetResult(None, None, None, None, 0, 0, 0)

    # cartesian batching only
    nvmap = (H.ndim - 2, 0, 0, 0, 0, 0, 0)
    f = cartesian_vmap(_floquet, in_axes, out_axes, nvmap)

    return f(H, T, tsave, solver, gradient, options)


def _floquet(
    H: TimeArray,
    T: Array,
    tsave: Array,
    solver: Solver,
    gradient: Gradient,
    options: Options,
) -> FloquetResult:
    # === check gradient is supported
    solver.assert_supports_gradient(gradient)

    # === integrator class is always FloquetIntegrator_t0
    integrator = FloquetIntegrator(
        ts=tsave, y0=None, H=H, solver=solver, gradient=gradient, options=options, T=T
    )

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_floquet_args(
    H: TimeArray, T: Array, tsave: Array
) -> (TimeArray, Array, Array):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check that tsave[-1] - tsave[0] <= T
    T, tsave = eqx.error_if(
        (T, tsave),
        tsave[-1] - tsave[0] > T,
        'The time interval spanned by tsave should be less than a single period T, '
        'i.e. `tsave[-1] - tsave[0] <= T`.',
    )

    # === check that the Hamiltonian is periodic with the supplied period
    H = eqx.error_if(
        H, H(0) != H(T), 'The Hamiltonian H is not periodic with the supplied period T.'
    )

    return H, T, tsave
