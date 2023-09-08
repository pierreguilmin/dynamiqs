from __future__ import annotations

from typing import Any

from .._utils import obj_type_str
from ..solvers.options import Dopri5, Euler, Rouchon1, Rouchon2
from ..solvers.result import Result
from ..solvers.utils import batch_H, batch_y0, check_time_tensor, to_td_tensor
from ..utils.tensor_types import ArrayLike, TDArrayLike, to_tensor
from ..utils.utils import is_ket, ket_to_dm
from .adaptive import MEDormandPrince5
from .euler import MEEuler
from .rouchon import MERouchon1, MERouchon2


def mesolve(
    H: TDArrayLike,
    jump_ops: list[ArrayLike],
    rho0: ArrayLike,
    t_save: ArrayLike,
    *,
    exp_ops: list[ArrayLike] | None = None,
    solver: str = 'dopri5',
    gradient: str | None = None,
    options: dict[str, Any] | None = None,
) -> Result:
    r"""Solve the Lindblad master equation.

    Evolve the density matrix $\rho(t)$ from an initial state $\rho(t=0) = \rho_0$
    according to the Lindblad master equation using a given Hamiltonian $H(t)$ and a
    list of jump operators $\{L_k\}$. The Lindblad master equation is given by

    $$
        \frac{d\rho}{dt} = -i[H, \rho] + \sum_k \left(L_k \rho L_k^\dagger -
        \frac{1}{2} \left \{L_k^\dagger L_k, \rho\right\}\right).
    $$

    For time-dependent problems, the Hamiltonian `H` can be passed as a function with
    signature `H(t: float) -> Tensor`. Extra Hamiltonian arguments and time-dependence
    for the jump operators are not yet supported.

    The Hamiltonian `H` and the initial density matrix `rho0` can be batched over to
    solve multiple master equations in a single run. The jump operators `jump_ops` and
    time list `t_save` are then common to all batches.

    `mesolve` can be differentiated through using either the default PyTorch autograd
    library (pass `gradient_alg="autograd"` in `options`), or a custom adjoint state
    differentiation (pass `gradient_alg="adjoint"` in `options`). By default, if no
    gradient is required, the graph of operations is not stored to improve performance.


    Args:
        H _(Tensor or Callable)_: Hamiltonian.
            Can be a tensor of shape `(n, n)` or `(b_H, n, n)` if batched, or a callable
            `H(t: float) -> Tensor` that returns a tensor of either possible shapes
            at every time between `t=0` and `t=t_save[-1]`.
        jump_ops _(Tensor, or list of Tensors)_: List of jump operators.
            Each jump operator should be a tensor of shape `(n, n)`.
        rho0 _(Tensor)_: Initial density matrix.
            Tensor of shape `(n, n)` or `(b_rho, n, n)` if batched.
        t_save _(Tensor, np.ndarray or list)_: Times for which results are saved.
            The master equation is solved from time `t=0.0` to `t=t_save[-1]`.
        exp_ops _(Tensor, or list of Tensors, optional)_: List of operators for which
            the expectation value is computed at every time value in `t_save`.
        solver _(str, optional)_: Solver to use. See the list of available solvers.
            Defaults to `"dopri5"`.
        gradient _(str, optional)_: Algorithm used for computing gradients.
            Can be either `"autograd"`, `"adjoint"` or `None`. Defaults to `None`.
        options _(dict, optional)_: Solver options. See the list of available
            solvers, and the options common to all solver below.

    Note-: Available solvers
      - `dopri5` --- Dormand-Prince method of order 5 (adaptive step). Default solver.
      - `euler` --- Euler method (fixed step).
      - `rouchon1` --- Rouchon method of order 1 (fixed step).
      - `rouchon2` --- Rouchon method of order 2 (fixed step).

    Note-: Available keys for `options`
        Common to all solvers:

        - **save_states** _(bool, optional)_ – If `True`, the state is saved at every
            time in `t_save`. If `False`, only the final state is stored and returned.
            Defaults to `True`.
        - **verbose** _(bool, optional)_ – If `True`, prints information about the
            integration progress. Defaults to `True`.
        - **dtype** _(torch.dtype, optional)_ – Complex data type to which all
            complex-valued tensors are converted. `t_save` is also converted to a real
            data type of the corresponding precision.
        - **device** _(torch.device, optional)_ – Device on which the tensors are
            stored.

        Required for `gradient="adjoint"`:

        - **parameters** _(tuple of nn.Parameter)_ – Parameters with respect to which
            the gradient is computed.

        Required for fixed step solvers (`euler`, `propagator`):

        - **dt** _(float)_ – Numerical time step of integration.

        Optional for adaptive step solvers (`dopri5`):

        - **atol** _(float, optional)_ – Absolute tolerance. Defaults to `1e-12`.
        - **rtol** _(float, optional)_ – Relative tolerance. Defaults to `1e-6`.
        - **max_steps** _(int, optional)_ – Maximum number of steps. Defaults to `1e6`.
        - **safety_factor** _(float, optional)_ – Safety factor in the step size
            prediction. Defaults to `0.9`.
        - **min_factor** _(float, optional)_ – Minimum factor by which the step size can
            decrease in a single step. Defaults to `0.2`.
        - **max_factor** _(float, optional)_ – Maximum factor by which the step size can
            increase in a single step. Defaults to `10.0`.

        Optional for `solver="rouchon1"`:

        - **sqrt_normalization** _(bool, optional)_ – If `True`, the Kraus map is
            renormalized at every step to preserve the trace of the density matrix.
            Only for time-independent problems. Ideal for stiff problems.
            Defaults to `False`.

    Warning: Warning for fixed step solvers
        For fixed time step solvers, the time list `t_save` should be strictly
        included in the time list used by the solver, given by `[0, dt, 2 * dt, ...]`
        where `dt` is defined with the `options` argument.

    Returns:
        Result of the master equation integration, as an instance of the `Result` class.
            The `result` object has the following attributes:

              - **y_save** or **states** _(Tensor)_ – Saved states.
              - **exp_save** or **expects** _(Tensor)_ – Saved expectation values.
              - **solver_str** (str): String representation of the solver.
              - **start_datetime** _(datetime)_ – Start time of the integration.
              - **end_datetime** _(datetime)_ – End time of the integration.
              - **total_time** _(datetime)_ – Total time of the integration.
              - **options** _(dict)_ – Solver options.
    """
    # H: (b_H?, n, n), rho0: (b_rho0?, n, n) -> (y_save, exp_save) with
    #    - y_save: (b_H?, b_rho0?, len(t_save), n, n)
    #    - exp_save: (b_H?, b_rho0?, len(exp_ops), len(t_save))

    # options
    if options is None:
        options = {}
    options['gradient_alg'] = gradient
    if solver == 'dopri5':
        options = Dopri5(**options)
        SOLVER_CLASS = MEDormandPrince5
    elif solver == 'euler':
        options = Euler(**options)
        SOLVER_CLASS = MEEuler
    elif solver == 'rouchon' or solver == 'rouchon1':
        options = Rouchon1(**options)
        SOLVER_CLASS = MERouchon1
    elif solver == 'rouchon2':
        options = Rouchon2(**options)
        SOLVER_CLASS = MERouchon2
    else:
        raise ValueError(f'Solver "{solver}" is not supported.')

    # check jump_ops
    if not isinstance(jump_ops, list):
        raise TypeError(
            'Argument `jump_ops` must be a list of array-like objects, but has type'
            f' {obj_type_str(jump_ops)}.'
        )
    if len(jump_ops) == 0:
        raise ValueError(
            'Argument `jump_ops` must be a non-empty list, otherwise consider using'
            ' `sesolve`.'
        )
    # check exp_ops
    if exp_ops is not None and not isinstance(exp_ops, list):
        raise TypeError(
            'Argument `exp_ops` must be `None` or a list of array-like objects, but'
            f' has type {obj_type_str(exp_ops)}.'
        )

    # format and batch all tensors
    # H: (b_H, 1, n, n)
    # rho0: (b_H, b_rho0, n, n)
    # exp_ops: (len(exp_ops), n, n)
    # jump_ops: (len(jump_ops), n, n)
    H = to_td_tensor(H, dtype=options.cdtype, device=options.device)
    rho0 = to_tensor(rho0, dtype=options.cdtype, device=options.device)
    H = batch_H(H)
    rho0 = batch_y0(rho0, H)
    if is_ket(rho0):
        rho0 = ket_to_dm(rho0)
    exp_ops = to_tensor(exp_ops, dtype=options.cdtype, device=options.device)
    jump_ops = to_tensor(jump_ops, dtype=options.cdtype, device=options.device)

    # convert t_save to a tensor
    t_save = to_tensor(t_save, dtype=options.rdtype, device=options.device)
    check_time_tensor(t_save, arg_name='t_save')

    # define the solver
    args = (H, rho0, t_save, exp_ops, options)
    solver = SOLVER_CLASS(*args, jump_ops=jump_ops)

    # compute the result
    solver.run()

    # get saved tensors and restore correct batching
    result = solver.result
    result.y_save = result.y_save.squeeze(1).squeeze(0)
    if result.exp_save is not None:
        result.exp_save = result.exp_save.squeeze(1).squeeze(0)

    return result
