from __future__ import annotations

import warnings
from functools import partial

import jax
import jax.numpy as jnp
from jaxtyping import Array, ArrayLike

from ..._checks import check_shape, check_times
from ...gradient import Gradient
from ...method import Expm, Method
from ...options import Options, check_options
from ...qarrays.dense_qarray import DenseQArray
from ...qarrays.qarray import QArrayLike
from ...result import MEPropagatorResult
from ...time_qarray import TimeQArray
from .._utils import (
    assert_method_supported,
    astimeqarray,
    cartesian_vmap,
    catch_xla_runtime_error,
    multi_vmap,
)
from ..core.expm_integrator import mepropagator_expm_integrator_constructor


def mepropagator(
    H: QArrayLike | TimeQArray,
    jump_ops: list[QArrayLike | TimeQArray],
    tsave: ArrayLike,
    *,
    method: Method = Expm(),  # noqa: B008
    gradient: Gradient | None = None,
    options: Options = Options(),  # noqa: B008
) -> MEPropagatorResult:
    r"""Compute the propagator of the Lindblad master equation.

    This function computes the propagator $\mathcal{U}(t)$ at time $t$ of the Lindblad
    master equation (with $\hbar=1$)
    $$
        \mathcal{U}(t) = \mathscr{T}\exp\left(\int_0^t\mathcal{L}(t')\dt'\right),
    $$
    where $\mathscr{T}$ is the time-ordering symbol and $\mathcal{L}$ is the system's
    Liouvillian. The formula simplifies to $\mathcal{U}(t)=e^{t\mathcal{L}}$ if the
    Liouvillian does not depend on time.

    Warning:
        This function only supports constant or piecewise constant Hamiltonians and jump
        operators. Support for arbitrary time dependence will be added soon.

    Args:
        H _(qarray-like or time-qarray of shape (...H, n, n))_: Hamiltonian.
        jump_ops _(list of qarray-like or time-qarray, each of shape (...Lk, n, n))_:
            List of jump operators.
        tsave _(array-like of shape (ntsave,))_: Times at which the propagators are
            saved. The equation is solved from `tsave[0]` to `tsave[-1]`, or from `t0`
            to `tsave[-1]` if `t0` is specified in `options`.
        method: Method for the integration. Defaults to
            [`dq.method.Expm`][dynamiqs.method.Expm] (explicit matrix exponentiation),
            which is the only supported method for now.
        gradient: Algorithm used to compute the gradient. The default is
            method-dependent, refer to the documentation of the chosen method for more
            details.
        options: Generic options (supported: `save_propagators`, `cartesian_batching`,
            `t0`, `save_extra`).
            ??? "Detailed options API"
                ```
                dq.Options(
                    save_propagators: bool = True,
                    cartesian_batching: bool = True,
                    t0: ScalarLike | None = None,
                    save_extra: callable[[Array], PyTree] | None = None,
                )
                ```

                **Parameters**

                - **save_propagators** - If `True`, the propagator is saved at every
                    time in `tsave`, otherwise only the final propagator is returned.
                - **cartesian_batching** - If `True`, batched arguments are treated as
                    separated batch dimensions, otherwise the batching is performed over
                    a single shared batched dimension.
                - **t0** - Initial time. If `None`, defaults to the first time in
                    `tsave`.
                - **save_extra** _(function, optional)_ - A function with signature
                    `f(QArray) -> PyTree` that takes a propagator as input and returns
                    a PyTree. This can be used to save additional arbitrary data
                    during the integration, accessible in `result.extra`.

    Returns:
        `dq.MEPropagatorResult` object holding the result of the propagator computation.
            Use `result.propagators` to access the saved propagators.

            ??? "Detailed result API"
                ```python
                dq.MEPropagatorResult
                ```

                **Attributes**

                - **propagators** _(qarray of shape (..., nsave, n^2, n^2))_ - Saved
                    propagators with `nsave = ntsave`, or `nsave = 1` if
                    `options.save_propagators=False`.
                - **final_propagator** _(qarray of shape (..., n^2, n^2))_ - Saved final
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

    ## Defining a time-dependent Hamiltonian or jump operator

    If the Hamiltonian or the jump operators depend on time, they can be converted
    to time-qarrays using [`dq.pwc()`][dynamiqs.pwc],
    [`dq.modulated()`][dynamiqs.modulated], or
    [`dq.timecallable()`][dynamiqs.timecallable]. See the
    [Time-dependent operators](../../documentation/basics/time-dependent-operators.md)
    tutorial for more details.

    ## Running multiple simulations concurrently

    The Hamiltonian `H` and the jump operators `jump_ops` can be batched to compute
    multiple propagators concurrently. All other arguments are common to every
    batch. The resulting propagators are batched according to the leading dimensions of
    `H` and `jump_ops`. The behaviour depends on the value of the `cartesian_batching`
    option.

    === "If `cartesian_batching = True` (default value)"
        The results leading dimensions are
        ```
        ... = ...H, ...L0, ...L1, (...)
        ```
        For example if:

        - `H` has shape _(2, 3, n, n)_,
        - `jump_ops = [L0, L1]` has shape _[(4, 5, n, n), (6, n, n)]_,

        then `result.propagators` has shape _(2, 3, 4, 5, 6, ntsave, n, n)_.
    === "If `cartesian_batching = False`"
        The results leading dimensions are
        ```
        ... = ...H = ...L0 = ...L1 = (...)  # (once broadcasted)
        ```
        For example if:

        - `H` has shape _(2, 3, n, n)_,
        - `jump_ops = [L0, L1]` has shape _[(3, n, n), (2, 1, n, n)]_,

        then `result.propagators` has shape _(2, 3, ntsave, n, n)_.

    See the
    [Batching simulations](../../documentation/basics/batching-simulations.md)
    tutorial for more details.
    """
    # === convert arguments
    H = astimeqarray(H)
    Ls = [astimeqarray(L) for L in jump_ops]
    tsave = jnp.asarray(tsave)

    # === check arguments
    _check_mepropagator_args(H, Ls)
    tsave = check_times(tsave, 'tsave')
    check_options(options, 'mepropagator')

    # we implement the jitted vectorization in another function to pre-convert QuTiP
    # objects (which are not JIT-compatible) to qarrays
    return _vectorized_mepropagator(H, Ls, tsave, method, gradient, options)


@catch_xla_runtime_error
@partial(jax.jit, static_argnames=('method', 'gradient', 'options'))
def _vectorized_mepropagator(
    H: TimeQArray,
    Ls: list[TimeQArray],
    tsave: Array,
    method: Method,
    gradient: Gradient | None,
    options: Options,
) -> MEPropagatorResult:
    # vectorize input over H and Ls
    in_axes = (H.in_axes, [L.in_axes for L in Ls], None, None, None, None)
    out_axes = MEPropagatorResult.out_axes()

    if options.cartesian_batching:
        nvmap = (H.ndim - 2, [L.ndim - 2 for L in Ls], 0, 0, 0, 0)
        f = cartesian_vmap(_mepropagator, in_axes, out_axes, nvmap)
    else:
        bshape = jnp.broadcast_shapes(*[x.shape[:-2] for x in [H, *Ls]])
        nvmap = len(bshape)
        # broadcast all vectorized input to same shape
        n = H.shape[-1]
        H = H.broadcast_to(*bshape, n, n)
        Ls = [L.broadcast_to(*bshape, n, n) for L in Ls]
        # vectorize the function
        f = multi_vmap(_mepropagator, in_axes, out_axes, nvmap)

    return f(H, Ls, tsave, method, gradient, options)


def _mepropagator(
    H: TimeQArray,
    Ls: list[TimeQArray],
    tsave: Array,
    method: Method,
    gradient: Gradient | None,
    options: Options,
) -> MEPropagatorResult:
    # === select integrator constructor
    integrator_constructors = {Expm: mepropagator_expm_integrator_constructor}
    assert_method_supported(method, integrator_constructors.keys())
    integrator_constructor = integrator_constructors[type(method)]

    # === check gradient is supported
    method.assert_supports_gradient(gradient)

    # === init integrator
    # todo: replace with vectorized utils constructor for eye
    data = jnp.eye(H.shape[-1] ** 2, dtype=H.dtype)
    # todo: timeqarray should expose dims without having to call at specific time
    y0 = DenseQArray(H(0.0).dims, True, data)
    integrator = integrator_constructor(
        ts=tsave,
        y0=y0,
        method=method,
        gradient=gradient,
        result_class=MEPropagatorResult,
        options=options,
        H=H,
        Ls=Ls,
    )

    # === run integrator
    result = integrator.run()

    # === return result
    return result  # noqa: RET504


def _check_mepropagator_args(H: TimeQArray, Ls: list[TimeQArray]):
    # === check H shape
    check_shape(H, 'H', '(..., n, n)', subs={'...': '...H'})

    # === check Ls shape
    for i, L in enumerate(Ls):
        check_shape(L, f'jump_ops[{i}]', '(..., n, n)', subs={'...': f'...L{i}'})

    if len(Ls) == 0:
        warnings.warn(
            'Argument `jump_ops` is an empty list, consider using `dq.sepropagator()`'
            ' to compute propagators for the Schrödinger equation.',
            stacklevel=2,
        )
