from __future__ import annotations

from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._checks import check_shape, check_times
from ...gradient import Gradient
from ...method import Dopri5, Dopri8, Euler, Kvaerno3, Kvaerno5, Method, Tsit5
from ...options import Options, check_options
from ...qarrays.qarray import QArrayLike
from ...result import FloquetResult
from ...time_qarray import TimeQArray
from .._utils import (
    assert_method_supported,
    astimeqarray,
    cartesian_vmap,
    catch_xla_runtime_error,
)
from ..core.floquet_integrator import floquet_integrator_constructor

__all__ = ['floquet']


def floquet(
    H: QArrayLike | TimeQArray,
    T: float,
    tsave: ArrayLike,
    *,
    method: Method = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> FloquetResult:
    r"""Compute Floquet modes and quasienergies of a periodic closed system.

    The Floquet modes $\Phi_{m}(t_0)$ and corresponding quasienergies $\epsilon_m$ are
    defined by the eigenvalue equation
    $$
        U(t_0, t_0+T)\Phi_{m}(t_0) = \exp(-i \epsilon_{m} T)\Phi_{m}(t_0),
    $$
    where $U(t_0, t_0+T)$ is the propagator over a single period $T$, with $t_0$ some
    arbitrary initial time (typically $t_0 = 0$). According to the Floquet theorem,
    these Floquet modes are periodic with period $T$ and form a complete basis for
    the time evolution of the system. The $t=t_0$ Floquet modes are obtained from
    diagonalization of the above propagator, while the $t \geq t_0$ Floquet modes are
    obtained by propagating the $t=t_0$ Floquet modes forward in time, via
    $$
        \Phi_{m}(t) = \exp(i\epsilon_{m}t)U(t_0, t_0+t)\Phi_{m}(t_0).
    $$

    Args:
        H _(qarray-like or time-qarray of shape (...H, n, n))_: Hamiltonian.
        T: Period of the Hamiltonian. If the Hamiltonian is batched, the period should
            be common over all elements in the batch. To batch over different periods,
            wrap the call to `floquet` in a `jax.vmap`, see above.
        tsave _(array-like of shape (ntsave,))_: Times at which to compute floquet
            modes. The specified times should be ordered, strictly ascending, and such
            that `tsave[-1] - tsave[0] <= T`.
        method: Method for the integration. Defaults to
            [`dq.method.Tsit5`][dynamiqs.method.Tsit5] (supported:
            [`Tsit5`][dynamiqs.method.Tsit5],
            [`Dopri5`][dynamiqs.method.Dopri5],
            [`Dopri8`][dynamiqs.method.Dopri8],
            [`Kvaerno3`][dynamiqs.method.Kvaerno3],
            [`Kvaerno5`][dynamiqs.method.Kvaerno5],
            [`Euler`][dynamiqs.method.Euler]).
        gradient: Algorithm used to compute the gradient.
        options: Generic options (supported: `progress_meter`, `t0`).
            ??? "Detailed options API"
                ```
                dq.Options(
                    progress_meter: AbstractProgressMeter | None = TqdmProgressMeter(),
                    t0: ScalarLike | None = None,
                )
                ```

                **Parameters**

                - **progress_meter** - Progress meter indicating how far the solve has
                    progressed. Defaults to a [tqdm](https://github.com/tqdm/tqdm)
                    progress meter. Pass `None` for no output, see other options in
                    [dynamiqs/progress_meter.py](https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/progress_meter.py).
                    If gradients are computed, the progress meter only displays during
                    the forward pass.
                - **t0** - Initial time. If `None`, defaults to the first time in
                    `tsave`.

    Returns:
        `dq.FloquetResult` object holding the result of the Floquet computation. Use
            `result.modes` to access the saved Floquet modes and `result.quasienergies`
            for the associated quasienergies.

            ??? "Detailed result API"
                ```python
                dq.FloquetResult
                ```

                **Attributes**

                - **modes** _(qarray of shape (..., ntsave, n, n, 1))_ - Saved Floquet
                    modes.
                - **quasienergies** _(array of shape (..., n))_ - Saved quasienergies
                - **T** _(float)_ - Drive period
                - **infos** _(PyTree or None)_ - Method-dependent information on the
                    resolution.
                - **tsave** _(array of shape (ntsave,))_ - Times for which results were
                    saved.
                - **method** _(Method)_ - Method used.
                - **gradient** _(Gradient)_ - Gradient used.
                - **options** _(Options)_ - Options used.

    # Advanced use-cases

    ## Running multiple simulations concurrently

    The Hamiltonian `H` can be batched to compute multiple Floquet modes and
    quasienergies concurrently. All other arguments are common to every batch. The
    Floquet modes and quasienergies are batched according to the leading dimensions of
    `H`. For example if `H` has shape _(2, 3, n, n)_, then `result.modes` has shape
    _(2, 3, ntsave, n, n, 1)_.

    See the
    [Batching simulations](../../documentation/basics/batching-simulations.md)
    tutorial for more details.

    ## Batching over drive periods

    The current API does not yet natively support batching over multiple drive
    periods, for instance if you wanted to batch over Hamiltonians with different
    drive frequencies. This however can be achieved straightforwardly with an
    external call to `jax.vmap`, as follows:

    ```python
    import jax
    import jax.numpy as jnp
    import dynamiqs as dq


    def single_floquet(omega):
        H = dq.modulated(lambda t: jnp.cos(omega * t), dq.sigmax())
        T = 2.0 * jnp.pi / omega
        tsave = jnp.linspace(0.0, T, 11)
        return dq.floquet(H, T, tsave)


    omegas = jnp.array([0.9, 1.0, 1.1])
    batched_floquet = jax.vmap(single_floquet)
    result = batched_floquet(omegas)
    ```
    """
    # === convert arguments
    H = astimeqarray(H)
    tsave = jnp.asarray(tsave)

    # === check arguments
    tsave = check_times(tsave, 'tsave')
    H, T, tsave = _check_floquet_args(H, T, tsave)
    check_options(options, 'floquet')

    # We implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to qarrays
    return _vectorized_floquet(H, T, tsave, method, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('method', 'gradient', 'options'))
def _vectorized_floquet(
    H: TimeQArray,
    T: float,
    tsave: Array,
    method: Method,
    gradient: Gradient,
    options: Options,
) -> FloquetResult:
    # vectorize input over H
    in_axes = (H.in_axes, None, None, None, None, None)
    out_axes = FloquetResult.out_axes()

    # cartesian batching only
    nvmap = (H.ndim - 2, 0, 0, 0, 0, 0, 0)
    f = cartesian_vmap(_floquet, in_axes, out_axes, nvmap)

    return f(H, T, tsave, method, gradient, options)


def _floquet(
    H: TimeQArray,
    T: float,
    tsave: Array,
    method: Method,
    gradient: Gradient,
    options: Options,
) -> FloquetResult:
    # === select integrator constructor
    supported_methods = (Tsit5, Dopri5, Dopri8, Kvaerno3, Kvaerno5, Euler)
    assert_method_supported(method, supported_methods)
    integrator_constructor = floquet_integrator_constructor

    # === check gradient is supported
    method.assert_supports_gradient(gradient)

    # === init integrator
    integrator = integrator_constructor(
        ts=tsave,
        y0=None,
        H=H,
        method=method,
        gradient=gradient,
        result_class=FloquetResult,
        options=options,
        T=T,
    )

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_floquet_args(
    H: TimeQArray, T: float, tsave: Array
) -> tuple[TimeQArray, float, Array]:
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
    # attach the check to `tsave` instead of `H` to workaround `CallableTimeQArray` that
    # do not have an underlying array to attach the check to
    rtol, atol = 1e-5, 1e-8  # TODO: fix hard-coded tolerance for periodicity check
    tsave = eqx.error_if(
        tsave,
        jnp.logical_not(
            eqx.tree_equal(H(0.0), H(T), rtol=rtol, atol=atol, typematch=True)
        ),
        'The Hamiltonian H is not periodic with the supplied period T.',
    )

    return H, T, tsave
