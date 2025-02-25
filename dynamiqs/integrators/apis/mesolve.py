from __future__ import annotations

import warnings
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike

from ..._checks import check_hermitian, check_shape, check_times
from ...gradient import Gradient
from ...method import (
    Dopri5,
    Dopri8,
    Euler,
    Expm,
    Kvaerno3,
    Kvaerno5,
    Method,
    Rouchon1,
    Tsit5,
)
from ...options import Options, check_options
from ...qarrays.qarray import QArray, QArrayLike
from ...qarrays.utils import asqarray
from ...result import MESolveResult
from ...time_qarray import TimeQArray
from .._utils import (
    assert_method_supported,
    astimeqarray,
    cartesian_vmap,
    catch_xla_runtime_error,
    multi_vmap,
)
from ..core.diffrax_integrator import (
    mesolve_dopri5_integrator_constructor,
    mesolve_dopri8_integrator_constructor,
    mesolve_euler_integrator_constructor,
    mesolve_kvaerno3_integrator_constructor,
    mesolve_kvaerno5_integrator_constructor,
    mesolve_tsit5_integrator_constructor,
)
from ..core.expm_integrator import mesolve_expm_integrator_constructor
from ..core.rouchon_integrator import mesolve_rouchon1_integrator_constructor


def mesolve(
    H: QArrayLike | TimeQArray,
    jump_ops: list[QArrayLike | TimeQArray],
    rho0: QArrayLike,
    tsave: ArrayLike,
    *,
    exp_ops: list[QArrayLike] | None = None,
    method: Method = Tsit5(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> MESolveResult:
    r"""Solve the Lindblad master equation.

    This function computes the evolution of the density matrix $\rho(t)$ at time $t$,
    starting from an initial state $\rho_0$, according to the Lindblad master
    equation (with $\hbar=1$ and where time is implicit(1))
    $$
        \frac{\dd\rho}{\dt} = -i[H, \rho]
        + \sum_{k=1}^N \left(
            L_k \rho L_k^\dag
            - \frac{1}{2} L_k^\dag L_k \rho
            - \frac{1}{2} \rho L_k^\dag L_k
        \right),
    $$
    where $H$ is the system's Hamiltonian and $\{L_k\}$ is a collection of jump
    operators.
    { .annotate }

    1. With explicit time dependence:
        - $\rho\to\rho(t)$
        - $H\to H(t)$
        - $L_k\to L_k(t)$

    Args:
        H _(qarray-like or time-qarray of shape (...H, n, n))_: Hamiltonian.
        jump_ops _(list of qarray-like or time-qarray, each of shape (...Lk, n, n))_:
            List of jump operators.
        rho0 _(qarray-like of shape (...rho0, n, 1) or (...rho0, n, n))_: Initial state.
        tsave _(array-like of shape (ntsave,))_: Times at which the states and
            expectation values are saved. The equation is solved from `tsave[0]` to
            `tsave[-1]`, or from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        exp_ops _(list of qarray-like, each of shape (n, n), optional)_: List of
            operators for which the expectation value is computed.
        method: Method for the integration. Defaults to
            [`dq.method.Tsit5`][dynamiqs.method.Tsit5] (supported:
            [`Tsit5`][dynamiqs.method.Tsit5], [`Dopri5`][dynamiqs.method.Dopri5],
            [`Dopri8`][dynamiqs.method.Dopri8],
            [`Kvaerno3`][dynamiqs.method.Kvaerno3],
            [`Kvaerno5`][dynamiqs.method.Kvaerno5],
            [`Euler`][dynamiqs.method.Euler],
            [`Rouchon1`][dynamiqs.method.Rouchon1],
            [`Expm`][dynamiqs.method.Expm]).
        gradient: Algorithm used to compute the gradient. The default is
            method-dependent, refer to the documentation of the chosen method for more
            details.
        options: Generic options (supported: `save_states`, `cartesian_batching`,
            `progress_meter`, `t0`, `save_extra`).
            ??? "Detailed options API"

                ```
                dq.Options(
                    save_states: bool = True,
                    cartesian_batching: bool = True,
                    progress_meter: AbstractProgressMeter | None = TqdmProgressMeter(),
                    t0: ScalarLike | None = None,
                    save_extra: callable[[Array], PyTree] | None = None,
                )
                ```

                **Parameters**

                - **save_states** - If `True`, the state is saved at every time in
                    `tsave`, otherwise only the final state is returned.
                - **cartesian_batching** - If `True`, batched arguments are treated as
                    separated batch dimensions, otherwise the batching is performed over
                    a single shared batched dimension.
                - **progress_meter** - Progress meter indicating how far the solve has
                    progressed. Defaults to a [tqdm](https://github.com/tqdm/tqdm)
                    progress meter. Pass `None` for no output, see other options in
                    [dynamiqs/progress_meter.py](https://github.com/dynamiqs/dynamiqs/blob/main/dynamiqs/progress_meter.py).
                    If gradients are computed, the progress meter only displays during
                    the forward pass.
                - **t0** - Initial time. If `None`, defaults to the first time in
                    `tsave`.
                - **save_extra** _(function, optional)_ - A function with signature
                    `f(QArray) -> PyTree` that takes a state as input and returns a
                    PyTree. This can be used to save additional arbitrary data
                    during the integration, accessible in `result.extra`.

    Returns:
        `dq.MESolveResult` object holding the result of the
            Lindblad master equation integration. Use `result.states` to access the
            saved states and `result.expects` to access the saved expectation values.

            ??? "Detailed result API"
                ```python
                dq.MESolveResult
                ```

                **Attributes**

                - **states** _(qarray of shape (..., nsave, n, n))_ - Saved states with
                    `nsave = ntsave`, or `nsave = 1` if `options.save_states=False`.
                - **final_state** _(qarray of shape (..., n, n))_ - Saved final state.
                - **expects** _(array of shape (..., len(exp_ops), ntsave) or None)_ - Saved
                    expectation values, if specified by `exp_ops`.
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

    ## Defining a time-dependent Hamiltonian or jump operator

    If the Hamiltonian or the jump operators depend on time, they can be converted to
    time-qarrays using [`dq.pwc()`][dynamiqs.pwc],
    [`dq.modulated()`][dynamiqs.modulated], or
    [`dq.timecallable()`][dynamiqs.timecallable]. See the
    [Time-dependent operators](../../documentation/basics/time-dependent-operators.md)
    tutorial for more details.

    ## Running multiple simulations concurrently

    The Hamiltonian `H`, the jump operators `jump_ops` and the initial density matrix
    `rho0` can be batched to solve multiple master equations concurrently. All other
    arguments are common to every batch. The resulting states and expectation values
    are batched according to the leading dimensions of `H`, `jump_ops` and  `rho0`. The
    behaviour depends on the value of the `cartesian_batching` option.

    === "If `cartesian_batching = True` (default value)"
        The results leading dimensions are
        ```
        ... = ...H, ...L0, ...L1, (...), ...rho0
        ```

        For example if:

        - `H` has shape _(2, 3, n, n)_,
        - `jump_ops = [L0, L1]` has shape _[(4, 5, n, n), (6, n, n)]_,
        - `rho0` has shape _(7, n, n)_,

        then `result.states` has shape _(2, 3, 4, 5, 6, 7, ntsave, n, n)_.
    === "If `cartesian_batching = False`"
        The results leading dimensions are
        ```
        ... = ...H = ...L0 = ...L1 = (...) = ...rho0  # (once broadcasted)
        ```

        For example if:

        - `H` has shape _(2, 3, n, n)_,
        - `jump_ops = [L0, L1]` has shape _[(3, n, n), (2, 1, n, n)]_,
        - `rho0` has shape _(3, n, n)_,

        then `result.states` has shape _(2, 3, ntsave, n, n)_.

    See the
    [Batching simulations](../../documentation/basics/batching-simulations.md)
    tutorial for more details.
    """  # noqa: E501
    # === convert arguments
    H = astimeqarray(H)
    Ls = [astimeqarray(L) for L in jump_ops]
    rho0 = asqarray(rho0)
    tsave = jnp.asarray(tsave)
    if exp_ops is not None:
        exp_ops = [asqarray(E) for E in exp_ops] if len(exp_ops) > 0 else None

    # === check arguments
    _check_mesolve_args(H, Ls, rho0, exp_ops)
    tsave = check_times(tsave, 'tsave')
    check_options(options, 'mesolve')

    # === convert rho0 to density matrix
    rho0 = rho0.todm()
    rho0 = check_hermitian(rho0, 'rho0')

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to qarrays
    return _vectorized_mesolve(H, Ls, rho0, tsave, exp_ops, method, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('method', 'gradient', 'options'))
def _vectorized_mesolve(
    H: TimeQArray,
    Ls: list[TimeQArray],
    rho0: QArray,
    tsave: Array,
    exp_ops: list[QArray] | None,
    method: Method,
    gradient: Gradient | None,
    options: Options,
) -> MESolveResult:
    # vectorize input over H, Ls and rho0
    in_axes = (H.in_axes, [L.in_axes for L in Ls], 0, None, None, None, None, None)
    out_axes = MESolveResult.out_axes()

    if options.cartesian_batching:
        nvmap = (H.ndim - 2, [L.ndim - 2 for L in Ls], rho0.ndim - 2, 0, 0, 0, 0, 0)
        f = cartesian_vmap(_mesolve, in_axes, out_axes, nvmap)
    else:
        bshape = jnp.broadcast_shapes(*[x.shape[:-2] for x in [H, *Ls, rho0]])
        nvmap = len(bshape)
        # broadcast all vectorized input to same shape
        n = H.shape[-1]
        H = H.broadcast_to(*bshape, n, n)
        Ls = [L.broadcast_to(*bshape, n, n) for L in Ls]
        rho0 = rho0.broadcast_to(*bshape, n, n)
        # vectorize the function
        f = multi_vmap(_mesolve, in_axes, out_axes, nvmap)

    return f(H, Ls, rho0, tsave, exp_ops, method, gradient, options)


def _mesolve(
    H: TimeQArray,
    Ls: list[TimeQArray],
    rho0: QArray,
    tsave: Array,
    exp_ops: list[QArray] | None,
    method: Method,
    gradient: Gradient | None,
    options: Options,
) -> MESolveResult:
    # === select integrator constructor
    integrator_constructors = {
        Euler: mesolve_euler_integrator_constructor,
        Rouchon1: mesolve_rouchon1_integrator_constructor,
        Dopri5: mesolve_dopri5_integrator_constructor,
        Dopri8: mesolve_dopri8_integrator_constructor,
        Tsit5: mesolve_tsit5_integrator_constructor,
        Kvaerno3: mesolve_kvaerno3_integrator_constructor,
        Kvaerno5: mesolve_kvaerno5_integrator_constructor,
        Expm: mesolve_expm_integrator_constructor,
    }
    assert_method_supported(method, integrator_constructors.keys())
    integrator_constructor = integrator_constructors[type(method)]

    # === check gradient is supported
    method.assert_supports_gradient(gradient)

    # === init integrator
    integrator = integrator_constructor(
        ts=tsave,
        y0=rho0,
        method=method,
        gradient=gradient,
        result_class=MESolveResult,
        options=options,
        H=H,
        Ls=Ls,
        Es=exp_ops,
    )

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_mesolve_args(
    H: TimeQArray, Ls: list[TimeQArray], rho0: QArray, exp_ops: list[QArray] | None
):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check Ls shape
    for i, L in enumerate(Ls):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)', subs={'...': f'...L{i}'})

    if len(Ls) == 0 and rho0.isket():
        warnings.warn(
            'Argument `jump_ops` is an empty list and argument `rho0` is a ket,'
            ' consider using `dq.sesolve()` to solve the Schrödinger equation.',
            stacklevel=2,
        )

    # === check rho0 shape
    check_shape(rho0, 'rho0', '(..., n, 1)', '(..., n, n)', subs={'...': '...rho0'})

    # === check exp_ops shape
    if exp_ops is not None:
        for i, E in enumerate(exp_ops):
            check_shape(E, f'exp_ops[{i}]', '(n, n)')
