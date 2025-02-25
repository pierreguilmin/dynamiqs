from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._checks import check_shape, check_times
from ...gradient import Gradient
from ...method import Dopri5, Dopri8, Euler, Expm, Kvaerno3, Kvaerno5, Method, Tsit5
from ...options import Options, check_options
from ...qarrays.layout import dense
from ...qarrays.qarray import QArrayLike
from ...result import SEPropagatorResult
from ...time_qarray import TimeQArray
from ...utils.operators import eye
from .._utils import (
    assert_method_supported,
    astimeqarray,
    cartesian_vmap,
    catch_xla_runtime_error,
    ispwc,
)
from ..core.diffrax_integrator import (
    sepropagator_dopri5_integrator_constructor,
    sepropagator_dopri8_integrator_constructor,
    sepropagator_euler_integrator_constructor,
    sepropagator_kvaerno3_integrator_constructor,
    sepropagator_kvaerno5_integrator_constructor,
    sepropagator_tsit5_integrator_constructor,
)
from ..core.expm_integrator import sepropagator_expm_integrator_constructor


def sepropagator(
    H: QArrayLike | TimeQArray,
    tsave: ArrayLike,
    *,
    method: Method | None = None,
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> SEPropagatorResult:
    r"""Compute the propagator of the Schrödinger equation.

    This function computes the propagator $U(t)$ at time $t$ of the Schrödinger
    equation (with $\hbar=1$)
    $$
        U(t) = \mathscr{T}\exp\left(-i\int_0^tH(t')\dt'\right),
    $$
    where $\mathscr{T}$ is the time-ordering symbol and $H$ is the system's
    Hamiltonian. The formula simplifies to $U(t)=e^{-iHt}$ if the Hamiltonian
    does not depend on time.

    If the Hamiltonian is constant or piecewise constant, the propagator is
    computed by directly exponentiating the Hamiltonian. Otherwise, the
    propagator is computed by solving the Schrödinger equation with an ODE method.

    Args:
        H _(qarray-like or time-qarray of shape (...H, n, n))_: Hamiltonian.
        tsave _(array-like of shape (ntsave,))_: Times at which the propagators
            are saved. The equation is solved from `tsave[0]` to `tsave[-1]`,
            or from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        method: Method for the integration. Defaults to `None` which redirects
            to [`dq.method.Expm`][dynamiqs.method.Expm] (explicit matrix
            exponentiation) or [`dq.method.Tsit5`][dynamiqs.method.Tsit5]
            depending on the Hamiltonian type (supported:
            [`Expm`][dynamiqs.method.Expm],
            [`Tsit5`][dynamiqs.method.Tsit5],
            [`Dopri5`][dynamiqs.method.Dopri5],
            [`Dopri8`][dynamiqs.method.Dopri8],
            [`Kvaerno3`][dynamiqs.method.Kvaerno3],
            [`Kvaerno5`][dynamiqs.method.Kvaerno5],
            [`Euler`][dynamiqs.method.Euler]).
        gradient: Algorithm used to compute the gradient. The default is
            method-dependent, refer to the documentation of the chosen method for more
            details.
        options: Generic options (supported: `save_propagators`, `progress_meter`, `t0`,
            `save_extra`).
            ??? "Detailed options API"
                ```
                dq.Options(
                    save_propagators: bool = True,
                    progress_meter: AbstractProgressMeter | None = TqdmProgressMeter(),
                    t0: ScalarLike | None = None,
                    save_extra: callable[[Array], PyTree] | None = None,
                )
                ```

                **Parameters**

                - **save_propagators** - If `True`, the propagator is saved at every
                    time in `tsave`, otherwise only the final propagator is returned.
                - **progress_meter** - Progress meter indicating how far the solve has
                    progressed. Defaults to a [tqdm](https://github.com/tqdm/tqdm)
                    progress meter. Pass `None` for no output, see other options in
                    [dynamiqs/progress_meter.py](https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/progress_meter.py).
                    If gradients are computed, the progress meter only displays during
                    the forward pass.
                - **t0** - Initial time. If `None`, defaults to the first time in
                    `tsave`.
                - **save_extra** _(function, optional)_ - A function with signature
                    `f(QArray) -> PyTree` that takes a propagator as input and returns
                    a PyTree. This can be used to save additional arbitrary data
                    during the integration, accessible in `result.extra`.


    Returns:
        `dq.SEPropagatorResult` object holding the result of the propagator computation.
            Use `result.propagators` to access the saved propagators.

            ??? "Detailed result API"
                ```python
                dq.SEPropagatorResult
                ```

                **Attributes**

                - **propagators** _(qarray of shape (..., nsave, n, n))_ - Saved
                    propagators with `nsave = ntsave`, or `nsave = 1` if
                    `options.save_propagators=False`.
                - **final_propagator** _(qarray of shape (..., n, n))_ - Saved final
                    propagator.
                - **extra** _(PyTree or None)_ - Extra data saved with `save_extra()` if
                    specified in `options`.
                - **infos** _(PyTree or None)_ - Method-dependent information on the
                    resolution.
                - **tsave** _(array of shape (ntsave,))_ - Times for which results were
                    saved.
                - **method** _(Method)_ - Method used.
                - **gradient** _(Gradient)_ - Gradient used.
                - **options** _(Options)_ - Options used.

    # Advanced use-cases

    ## Defining a time-dependent Hamiltonian

    If the Hamiltonian depends on time, it can be converted to a time-qarray using
    [`dq.pwc()`][dynamiqs.pwc], [`dq.modulated()`][dynamiqs.modulated], or
    [`dq.timecallable()`][dynamiqs.timecallable]. See the
    [Time-dependent operators](../../documentation/basics/time-dependent-operators.md)
    tutorial for more details.

    ## Running multiple simulations concurrently

    The Hamiltonian `H` can be batched to compute multiple propagators
    concurrently. All other arguments are common to every batch. The resulting
    propagators are batched according to the leading dimensions of `H`. For example if
    `H` has shape _(2, 3, n, n)_, then `result.propagators` has shape
    _(2, 3, ntsave, n, n)_.

    See the
    [Batching simulations](../../documentation/basics/batching-simulations.md)
    tutorial for more details.
    """
    # === convert arguments
    H = astimeqarray(H)
    tsave = jnp.asarray(tsave)

    # === check arguments
    _check_sepropagator_args(H)
    tsave = check_times(tsave, 'tsave')
    check_options(options, 'sepropagator')

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to qarrays
    return _vectorized_sepropagator(H, tsave, method, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('method', 'gradient', 'options'))
def _vectorized_sepropagator(
    H: TimeQArray,
    tsave: Array,
    method: Method,
    gradient: Gradient | None,
    options: Options,
) -> SEPropagatorResult:
    # vectorize input over H
    in_axes = (H.in_axes, None, None, None, None)
    out_axes = SEPropagatorResult.out_axes()

    # cartesian batching only
    nvmap = (H.ndim - 2, 0, 0, 0, 0, 0)
    f = cartesian_vmap(_sepropagator, in_axes, out_axes, nvmap)

    return f(H, tsave, method, gradient, options)


def _sepropagator(
    H: TimeQArray,
    tsave: Array,
    method: Method | None,
    gradient: Gradient | None,
    options: Options,
) -> SEPropagatorResult:
    # === select integrator constructor
    if method is None:  # default method
        method = Expm() if ispwc(H) else Tsit5()

    integrator_constructors = {
        Expm: sepropagator_expm_integrator_constructor,
        Euler: sepropagator_euler_integrator_constructor,
        Dopri5: sepropagator_dopri5_integrator_constructor,
        Dopri8: sepropagator_dopri8_integrator_constructor,
        Tsit5: sepropagator_tsit5_integrator_constructor,
        Kvaerno3: sepropagator_kvaerno3_integrator_constructor,
        Kvaerno5: sepropagator_kvaerno5_integrator_constructor,
    }
    assert_method_supported(method, integrator_constructors.keys())
    integrator_constructor = integrator_constructors[type(method)]

    # === check gradient is supported
    method.assert_supports_gradient(gradient)

    # === init integrator
    y0 = eye(*H.dims, layout=dense)
    integrator = integrator_constructor(
        ts=tsave,
        y0=y0,
        method=method,
        gradient=gradient,
        result_class=SEPropagatorResult,
        options=options,
        H=H,
    )

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_sepropagator_args(H: TimeQArray):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})
