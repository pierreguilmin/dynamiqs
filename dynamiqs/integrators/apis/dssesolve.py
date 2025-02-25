from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike, PRNGKeyArray

from ..._checks import check_shape, check_times
from ...gradient import Gradient
from ...method import EulerMaruyama, Method
from ...options import Options, check_options
from ...qarrays.qarray import QArray, QArrayLike
from ...qarrays.utils import asqarray
from ...result import DSSESolveResult
from ...time_qarray import TimeQArray
from .._utils import (
    assert_method_supported,
    astimeqarray,
    cartesian_vmap,
    catch_xla_runtime_error,
    multi_vmap,
)
from ..core.fixed_step_stochastic_integrator import (
    dssesolve_euler_maruyama_integrator_constructor,
)


def dssesolve(
    H: QArrayLike | TimeQArray,
    jump_ops: list[QArrayLike | TimeQArray],
    psi0: QArrayLike,
    tsave: ArrayLike,
    keys: PRNGKeyArray,
    *,
    exp_ops: list[QArrayLike] | None = None,
    method: Method | None = None,
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> DSSESolveResult:
    r"""Solve the diffusive stochastic Schrödinger equation (SSE).

    The diffusive SSE describes the evolution of a quantum system measured
    by an ideal diffusive detector (for example homodyne or heterodyne detection
    in quantum optics). This function computes the evolution of the state vector
    $\ket{\psi(t)}$ at time $t$, starting from an initial state $\ket{\psi_0}$
    according to the diffusive SSE in Itô form ($\hbar=1$, time is implicit(1))
    $$
        \begin{split}
            \dd\\!\ket\psi = \Bigg[
                &-iH \dt
                -\frac12 \sum_{k=1}^N \left(
                    L_k^\dag L_k - \braket{L_k + L_k^\dag} L_k
                    + \frac14 \braket{L_k + L_k^\dag }^2
                \right) \dt \\\\
                &+ \sum_{k=1}^N \left(
                    L_k - \frac12 \braket{L_k + L_k^\dag}
                \right) \dd W_k
            \Bigg] \\!\ket\psi
        \end{split}
    $$
    where $H$ is the system's Hamiltonian, $\{L_k\}$ is a collection of jump operators,
    each continuously measured with perfect efficiency and $\dd W_k$ are independent
    Wiener processes.
    { .annotate }

    1. With explicit time dependence:
        - $\ket\psi\to\ket{\psi(t)}$
        - $H\to H(t)$
        - $L_k\to L_k(t)$
        - $\dd W_k\to \dd W_k(t)$

    The continuous-time measurements are defined with the Itô processes $\dd Y_k$ (time
    is implicit)
    $$
        \dd Y_k = \braket{L_k + L_k^\dag} \dt + \dd W_k.
    $$

    The solver returns the time-averaged measurements $I_k(t_n, t_{n+1})$ defined for
    each time interval $[t_n, t_{n+1}[$ by
    $$
        I_k(t_n, t_{n+1}) = \frac{Y_k(t_{n+1}) - Y_k(t_n)}{t_{n+1} - t_n}
        = \frac{1}{t_{n+1}-t_n}\int_{t_n}^{t_{n+1}} \dd Y_k(t)
    $$
    The time intervals $[t_n, t_{n+1}[$ are defined by `tsave`, so the number of
    returned measurement values for each detector is `len(tsave)-1`.

    Warning:
        For now, `dssesolve()` only supports linearly spaced `tsave` with values that
        are exact multiples of the method fixed step size `dt`.

    Args:
        H _(qarray-like or time-qarray of shape (...H, n, n))_: Hamiltonian.
        jump_ops _(list of qarray-like or time-qarray, each of shape (n, n))_: List of
            jump operators.
        psi0 _(qarray-like of shape (...psi0, n, 1))_: Initial state.
        tsave _(array-like of shape (ntsave,))_: Times at which the states and
            expectation values are saved. The equation is solved from `tsave[0]` to
            `tsave[-1]`. Measurements are time-averaged and saved over each interval
            defined by `tsave`.
        keys _(list of PRNG keys)_: PRNG keys used to sample the Wiener processes.
            The number of elements defines the number of sampled stochastic
            trajectories.
        exp_ops _(list of array-like, each of shape (n, n), optional)_: List of
            operators for which the expectation value is computed.
        method: Method for the integration. No defaults for now, you have to specify a
            method (supported: [`EulerMaruyama`][dynamiqs.method.EulerMaruyama]).
        gradient: Algorithm used to compute the gradient. The default is
            method-dependent, refer to the documentation of the chosen method for more
            details.
        options: Generic options (supported: `save_states`, `cartesian_batching`,
            `save_extra`).
            ??? "Detailed options API"
                ```
                dq.Options(
                    save_states: bool = True,
                    cartesian_batching: bool = True,
                    save_extra: callable[[Array], PyTree] | None = None,
                )
                ```

                **Parameters**

                - **save_states** - If `True`, the state is saved at every time in
                    `tsave`, otherwise only the final state is returned.
                - **cartesian_batching** - If `True`, batched arguments are treated as
                    separated batch dimensions, otherwise the batching is performed over
                    a single shared batched dimension.
                - **save_extra** _(function, optional)_ - A function with signature
                    `f(QArray) -> PyTree` that takes a state as input and returns a
                    PyTree. This can be used to save additional arbitrary data
                    during the integration, accessible in `result.extra`.

    Returns:
        `dq.DSSESolveResult` object holding the result of the diffusive SSE integration.
            Use `result.states` to access the saved states, `result.expects` to access
            the saved expectation values and `result.measurements` to access the
            detector measurements.

            ??? "Detailed result API"
                ```python
                dq.DSSESolveResult
                ```

                For the shape indications we define `ntrajs` as the number of trajectories
                (`ntrajs = len(keys)`).

                **Attributes**

                - **states** _(qarray of shape (..., ntrajs, nsave, n, 1))_ - Saved
                    states with `nsave = ntsave`, or `nsave = 1` if
                    `options.save_states=False`.
                - **final_state** _(qarray of shape (..., ntrajs, n, 1))_ - Saved final
                    state.
                - **expects** _(array of shape (..., ntrajs, len(exp_ops), ntsave) or None)_ - Saved
                    expectation values, if specified by `exp_ops`.
                - **measurements** _(array of shape (..., ntrajs, len(jump_ops), nsave-1))_ - Saved
                    measurements.
                - **extra** _(PyTree or None)_ - Extra data saved with `save_extra()` if
                    specified in `options`.
                - **keys** _(PRNG key array of shape (ntrajs,))_ - PRNG keys used to
                    sample the Wiener processes.
                - **infos** _(PyTree or None)_ - Method-dependent information on the
                    resolution.
                - **tsave** _(array of shape (ntsave,))_ - Times for which results were
                    saved.
                - **method** _(Method)_ - Method used.
                - **gradient** _(Gradient)_ - Gradient used.
                - **options** _(Options)_ - Options used.

    # Advanced use-cases

    ## Defining a time-dependent Hamiltonian or jump operator

    If the Hamiltonian or the jump operators depend on time, they can be converted
    to time-arrays using [`dq.pwc()`][dynamiqs.pwc],
    [`dq.modulated()`][dynamiqs.modulated], or
    [`dq.timecallable()`][dynamiqs.timecallable]. See the
    [Time-dependent operators](../../documentation/basics/time-dependent-operators.md)
    tutorial for more details.

    ## Running multiple simulations concurrently

    The Hamiltonian `H` and the initial state `psi0` can be batched to
    solve multiple SSEs concurrently. All other arguments (including the PRNG key)
    are common to every batch. The resulting states, measurements and expectation values
    are batched according to the leading dimensions of `H` and `psi0`. The
    behaviour depends on the value of the `cartesian_batching` option.

    === "If `cartesian_batching = True` (default value)"
        The results leading dimensions are
        ```
        ... = ...H, ...psi0
        ```
        For example if:

        - `H` has shape _(2, 3, n, n)_,
        - `psi0` has shape _(4, n, 1)_,

        then `result.states` has shape _(2, 3, 4, ntrajs, ntsave, n, 1)_.

    === "If `cartesian_batching = False`"
        The results leading dimensions are
        ```
        ... = ...H = ...psi0  # (once broadcasted)
        ```
        For example if:

        - `H` has shape _(2, 3, n, n)_,
        - `psi0` has shape _(3, n, 1)_,

        then `result.states` has shape _(2, 3, ntrajs, ntsave, n, 1)_.

    See the
    [Batching simulations](../../documentation/basics/batching-simulations.md)
    tutorial for more details.

    Warning:
        Batching on `jump_ops` is not yet supported, if this is needed don't
        hesitate to
        [open an issue on GitHub](https://github.com/dynamiqs/dynamiqs/issues/new).
    """  # noqa: E501
    # === convert arguments
    H = astimeqarray(H)
    Ls = [astimeqarray(L) for L in jump_ops]
    psi0 = asqarray(psi0)
    tsave = jnp.asarray(tsave)
    keys = jnp.asarray(keys)
    if exp_ops is not None:
        exp_ops = [asqarray(E) for E in exp_ops] if len(exp_ops) > 0 else None

    # === check arguments
    _check_dssesolve_args(H, Ls, psi0, exp_ops)
    tsave = check_times(tsave, 'tsave')
    check_options(options, 'dssesolve')

    if method is None:
        raise ValueError('Argument `method` must be specified.')

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to JAX arrays
    tsave = tuple(tsave.tolist())  # todo: fix static tsave
    return _vectorized_dssesolve(
        H, Ls, psi0, tsave, keys, exp_ops, method, gradient, options
    )


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('tsave', 'method', 'gradient', 'options'))
def _vectorized_dssesolve(
    H: TimeQArray,
    Ls: list[TimeQArray],
    psi0: QArray,
    tsave: Array,
    keys: PRNGKeyArray,
    exp_ops: list[QArray] | None,
    method: Method,
    gradient: Gradient | None,
    options: Options,
) -> DSSESolveResult:
    f = _dssesolve_single_trajectory

    # === vectorize function over stochastic trajectories
    # the input is vectorized over `key`
    in_axes = (None, None, None, None, 0, None, None, None, None)
    # the result is vectorized over `_saved`, `infos` and `keys`
    out_axes = DSSESolveResult.out_axes()
    f = jax.vmap(f, in_axes, out_axes)

    # === vectorize function
    # vectorize input over H and psi0
    in_axes = (H.in_axes, None, 0, None, None, None, None, None, None)

    if options.cartesian_batching:
        nvmap = (H.ndim - 2, 0, psi0.ndim - 2, 0, 0, 0, 0, 0, 0)
        f = cartesian_vmap(f, in_axes, out_axes, nvmap)
    else:
        n = H.shape[-1]
        bshape = jnp.broadcast_shapes(H.shape[:-2], psi0.shape[:-2])
        nvmap = len(bshape)
        # broadcast all vectorized input to same shape
        H = H.broadcast_to(*bshape, n, n)
        psi0 = psi0.broadcast_to(*bshape, n, 1)
        # vectorize the function
        f = multi_vmap(f, in_axes, out_axes, nvmap)

    # === apply vectorized function
    return f(H, Ls, psi0, tsave, keys, exp_ops, method, gradient, options)


def _dssesolve_single_trajectory(
    H: TimeQArray,
    Ls: list[TimeQArray],
    psi0: QArray,
    tsave: Array,
    key: PRNGKeyArray,
    exp_ops: list[QArray] | None,
    method: Method,
    gradient: Gradient | None,
    options: Options,
) -> DSSESolveResult:
    # === select integrator constructor
    integrator_constructors = {
        EulerMaruyama: dssesolve_euler_maruyama_integrator_constructor
    }
    assert_method_supported(method, integrator_constructors.keys())
    integrator_constructor = integrator_constructors[type(method)]

    # === check gradient is supported
    method.assert_supports_gradient(gradient)

    # === init integrator
    integrator = integrator_constructor(
        ts=tsave,
        y0=psi0,
        method=method,
        gradient=gradient,
        result_class=DSSESolveResult,
        options=options,
        H=H,
        Ls=Ls,
        Es=exp_ops,
        key=key,
    )

    # === run solver
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_dssesolve_args(
    H: TimeQArray, Ls: list[TimeQArray], psi0: QArray, exp_ops: list[QArray] | None
):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check Ls shape
    for i, L in enumerate(Ls):
        check_shape(L, f'jump_ops[{i}]', '(n, n)')

    # === check psi0 shape
    check_shape(psi0, 'psi0', '(..., n, 1)', subs={'...': '...psi0'})

    # === check exp_ops shape
    if exp_ops is not None:
        for i, E in enumerate(exp_ops):
            check_shape(E, f'exp_ops[{i}]', '(n, n)')
