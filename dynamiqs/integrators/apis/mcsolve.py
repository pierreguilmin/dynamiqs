from __future__ import annotations

import logging
from functools import partial

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import ArrayLike
from optimistix import AbstractRootFinder

from ..._checks import check_shape, check_times
from ...gradient import Gradient
from ...method import Dopri5, Dopri8, Euler, Kvaerno3, Kvaerno5, Method, Tsit5
from ...options import Options
from ...qarrays.qarray import QArray, QArrayLike
from ...qarrays.utils import asqarray
from ...result import MCSolveResult
from ...time_qarray import TimeQArray
from .._utils import (
    _astimeqarray,
    assert_method_supported,
    cartesian_vmap,
    catch_xla_runtime_error,
    multi_vmap,
)
from ..core.diffrax_integrator import (
    mcsolve_dopri5_integrator_constructor,
    mcsolve_dopri8_integrator_constructor,
    mcsolve_euler_integrator_constructor,
    mcsolve_kvaerno3_integrator_constructor,
    mcsolve_kvaerno5_integrator_constructor,
    mcsolve_tsit5_integrator_constructor,
)


def mcsolve(
    H: QArrayLike | TimeQArray,
    jump_ops: list[QArrayLike | TimeQArray],
    psi0: QArrayLike,
    tsave: ArrayLike,
    keys: ArrayLike,
    *,
    exp_ops: list[QArrayLike] | None = None,
    method: Method = Tsit5(),  # noqa: B008
    root_finder: AbstractRootFinder | None = None,
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> MCSolveResult:
    r"""Solve the Lindblad master equation through Monte-Carlo sampling of its jump unraveling.

    This function computes the evolution of the density matrix $\rho(t)$ at time $t$,
    starting from an initial state vector $\psi_0$, according to the Monte-Carlo jump unraveling
    of the Lindblad master equation. In this unraveling, a state vector evolves in between jumps
    with with non-Hermitian Hamiltonian ($\hbar=1$)
    $$
        \frac{\dd\ket{\psi(t)}}{\dt}
        = -i [H(t) -\frac{i}{2} \sum_{k=1}^{N} L_{k}^{\dagger} (t)L_{k}(t)] \ket{\psi(t)},
    $$
    where $H(t)$ is the system's Hamiltonian at time $t$ and $\{L_k(t)\}$ is a
    collection of jump operators at time $t$. The norm of the state is tracked, and jumps
    are applied when the norm of the state falls below that of a random number.
    We follow the algorithm outlined in [Abdelhafez et al. (2019)](https://journals.aps.org/pra/abstract/10.1103/PhysRevA.99.052327)
    to efficiently perform this Monte-Carlo sampling: first the no-jump trajectory is
    computed, and we extract the norm of the resulting final state. This norm is then
    taken as a lower bound for random numbers sampled for the jump trajectories, such
    that they all experience at least one jump.

    Quote: Time-dependent Hamiltonian or jump operators
        If the Hamiltonian or the jump operators depend on time, they can be converted
        to time-qarrays using [`dq.constant()`](/python_api/time_qarray/constant.html),
        [`dq.pwc()`](/python_api/time_qarray/pwc.html),
        [`dq.modulated()`](/python_api/time_qarray/modulated.html), or
        [`dq.timecallable()`](/python_api/time_qarray/timecallable.html).

    Quote: Running multiple simulations concurrently
        The Hamiltonian `H`, the jump operators `Ls` the initial state `psi0` and the
        `keys` can be batched to solve multiple monte-carlo equations concurrently. All
        other arguments are common to every batch.

    Args:
        H _(qarray-like or time-array of shape (...H, n, n))_: Hamiltonian.
        Ls _(list of qarray-like or time-array, of shape (nL, n, n))_: List of
            jump operators.
        psi0 _(qarray-like of shape (...psi0, n, 1))_: Initial state.
        tsave _(array-like of shape (nt,))_: Times at which the states and expectation
            values are saved. The equation is solved from `tsave[0]` to `tsave[-1]`, or
            from `t0` to `tsave[-1]` if `t0` is specified in `options`.
        keys _(Array of shape (...ntraj,))_: PRNG keys used to sample the jump trajectories.
            With options.cartesian_batching=True, this array must be 1D and the number of
            elements defines the number of sampled trajectories, not including the no-jump
            trajectory. With options.cartesian_batching=False, the leading dimensions
            of keys must be broadcastable with those of H, Ls, psi0, with the final
            dimension determining the number of trajectories.
        exp_ops _(list of qarray-like, of shape (nE, n, n), optional)_: List of
            operators for which the expectation value is computed.
        method: Method for the integration. Defaults to
            [`dq.method.Tsit5()`](/python_api/method/Tsit5.html).
        root_finder: Root finder passed to dx.diffeqsolve() (see [here](https://docs.kidger.site/diffrax/api/diffeqsolve/))
            to find the exact time an event occurs. Can be `None`, in which case the
            root finding functionality is not utilized. It is recommended to pass a root
            finder (such as the optimistix Newton root finder, see [here](https://docs.kidger.site/optimistix/api/root_find/#optimistix.Newton))
            so that event times are not determined by the integration step sizes in
            diffeqsolve. However there are cases where the root finding can fail,
            causing the whole simulation to fail. Passing `None` for `root_finder` can
            alleviate the issue in these cases.
        gradient: Algorithm used to compute the gradient.
        options: Generic options, see [`dq.Options`](/python_api/options/Options.html).

    Returns:
        [`dq.Result`](/python_api/result/Result.html) object holding the result of the
            Monte-Carlo integration. Use the attributes `no_jump_states`, `jump_states`
            and `expects` to access saved quantities, more details in
            [`dq.MCSolveResult`][dynamiqs.MCSolveResult].
    """  # noqa E501
    # === convert arguments
    H = _astimeqarray(H)
    Ls = [_astimeqarray(L) for L in jump_ops]
    psi0 = asqarray(psi0)
    tsave = jnp.asarray(tsave)
    keys = jnp.asarray(keys)
    if exp_ops is not None:
        exp_ops = [asqarray(E) for E in exp_ops] if len(exp_ops) > 0 else None

    # === check arguments
    _check_mcsolve_args(H, Ls, psi0, exp_ops)
    tsave = check_times(tsave, 'tsave')

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to JAX arrays
    return _vectorized_mcsolve(
        H, Ls, psi0, tsave, keys, exp_ops, method, root_finder, gradient, options
    )


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=['method', 'root_finder', 'gradient', 'options'])
def _vectorized_mcsolve(
    H: TimeQArray,
    Ls: list[TimeQArray],
    psi0: QArray,
    tsave: Array,
    keys: Array,
    exp_ops: list[QArray] | None,
    method: Method,
    root_finder: AbstractRootFinder,
    gradient: Gradient | None,
    options: Options,
) -> MCSolveResult:
    # vectorize output over `_no_jump_res`, `_jump_res`, `no_jump_prob`, `jump_times`,
    # `num_jumps`
    out_axes = MCSolveResult.out_axes()

    if options.cartesian_batching:
        # vectorize input over H, Ls, psi0.
        in_axes = (
            H.in_axes,
            [L.in_axes for L in Ls],
            0,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        )
        nvmap = (
            H.ndim - 2,
            [L.ndim - 2 for L in Ls],
            psi0.ndim - 2,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
        )
        f = cartesian_vmap(_mcsolve, in_axes, out_axes, nvmap)
    else:
        # vectorize input over H, Ls, psi0 and keys.
        in_axes = (
            H.in_axes,
            [L.in_axes for L in Ls],
            0,
            None,
            0,
            None,
            None,
            None,
            None,
            None,
        )
        shapes = [x.shape[:-2] for x in [H, *Ls, psi0]] + [keys.shape[:-1]]
        bshape = jnp.broadcast_shapes(*shapes)
        nvmap = len(bshape)
        # broadcast all vectorized input to same shape
        n = H.shape[-1]
        H = H.broadcast_to(*bshape, n, n)
        Ls = [L.broadcast_to(*bshape, n, n) for L in Ls]
        psi0 = psi0.broadcast_to(*bshape, n, 1)
        keys = jnp.broadcast_to(keys, (*bshape, keys.shape[-1]))
        # vectorize the function
        f = multi_vmap(_mcsolve, in_axes, out_axes, nvmap)

    return f(H, Ls, psi0, tsave, keys, exp_ops, method, root_finder, gradient, options)


def _mcsolve(
    H: TimeQArray,
    Ls: list[TimeQArray],
    psi0: QArray,
    tsave: Array,
    keys: Array,
    exp_ops: list[QArray] | None,
    method: Method,
    root_finder: AbstractRootFinder,
    gradient: Gradient | None,
    options: Options,
) -> MCSolveResult:
    integrator_constructors = {
        Euler: mcsolve_euler_integrator_constructor,
        Dopri5: mcsolve_dopri5_integrator_constructor,
        Dopri8: mcsolve_dopri8_integrator_constructor,
        Tsit5: mcsolve_tsit5_integrator_constructor,
        Kvaerno3: mcsolve_kvaerno3_integrator_constructor,
        Kvaerno5: mcsolve_kvaerno5_integrator_constructor,
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
        result_class=MCSolveResult,
        options=options,
        H=H,
        Ls=Ls,
        keys=keys,
        root_finder=root_finder,
        Es=exp_ops,
    )

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_mcsolve_args(
    H: TimeQArray, Ls: list[TimeQArray], psi0: QArray, exp_ops: list[QArray] | None
):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check Ls shape
    for i, L in enumerate(Ls):
        check_shape(L, f'Ls[{i}]', '(..., n, n)', subs={'...': f'...L{i}'})

    if len(Ls) == 0:
        logging.warning(
            'Argument `jump_ops` is an empty list, consider using `dq.sesolve()` to'
            ' solve the Schrödinger equation.'
        )

    # === check psi0 shape
    check_shape(psi0, 'psi0', '(..., n, 1)', subs={'...': '...psi0'})

    # === check exp_ops shape
    if exp_ops is not None:
        for i, E in enumerate(exp_ops):
            check_shape(E, f'exp_ops[{i}]', '(n, n)')
