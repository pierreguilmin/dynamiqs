from __future__ import annotations

import torch

from ..options import Euler, Options
from ..solvers.result import Result
from ..solvers.utils.tensor_formatter import TensorFormatter
from ..solvers.utils.utils import check_time_tensor
from ..utils.tensor_types import OperatorLike, TDOperatorLike, TensorLike
from ..utils.utils import obj_type_str
from .euler import SMEEuler


def smesolve(
    H: TDOperatorLike,
    jump_ops: list[OperatorLike],
    rho0: OperatorLike,
    t_save: TensorLike,
    etas: TensorLike,
    ntrajs: int,
    *,
    t_meas: TensorLike | None = None,
    seed: int | None = None,
    exp_ops: list[OperatorLike] | None = None,
    options: Options | None = None,
) -> Result:
    """Solve the Stochastic master equation."""
    # H: (b_H?, n, n), rho0: (b_rho0?, n, n) -> (y_save, exp_save, meas_save) with
    #    - y_save: (b_H?, b_rho0?, ntrajs, len(t_save), n, n)
    #    - exp_save: (b_H?, b_rho0?, ntrajs, len(exp_ops), len(t_save))
    #    - meas_save: (b_H?, b_rho0?, ntrajs, len(meas_ops), len(t_meas) - 1)

    # default options
    if options is None:
        raise ValueError(
            'No default solver yet, please specify one using the `options` argument.'
        )

    # check jump_ops
    if not isinstance(jump_ops, list):
        raise TypeError(
            'Argument `jump_ops` must be a list of array-like objects, but has type'
            f' {obj_type_str(jump_ops)}.'
        )
    if len(jump_ops) == 0:
        raise ValueError(
            'Argument `jump_ops` must be a non-empty list, otherwise consider using'
            ' `ssesolve`.'
        )

    # check exp_ops
    if exp_ops is not None and not isinstance(exp_ops, list):
        raise TypeError(
            'Argument `exp_ops` must be `None` or a list of array-like objects, but'
            f' has type {obj_type_str(exp_ops)}.'
        )

    # format and batch all tensors
    # H_batched: (b_H, 1, 1, n, n)
    # rho0_batched: (b_H, b_rho0, ntrajs, n, n)
    formatter = TensorFormatter(options.cdtype, options.device)
    H_batched, rho0_batched = formatter.format_H_and_state(H, rho0, state_to_dm=True)
    H_batched = H_batched.unsqueeze(2)
    rho0_batched = rho0_batched.unsqueeze(2).repeat(1, 1, ntrajs, 1, 1)
    exp_ops = formatter.format(exp_ops)  # (len(exp_ops), n, n)
    jump_ops = formatter.format(jump_ops)  # (len(jump_ops), n, n)

    # convert t_save to a tensor
    t_save = torch.as_tensor(t_save, dtype=options.rdtype, device=options.device)
    check_time_tensor(t_save, arg_name='t_save')

    # convert etas to a tensor and check
    etas = torch.as_tensor(etas, dtype=options.rdtype, device=options.device)
    if len(etas) != len(jump_ops):
        raise ValueError(
            'Argument `etas` must have the same length as `jump_ops` of length'
            f' {len(jump_ops)}, but has length {len(etas)}.'
        )
    if torch.all(etas == 0.0):
        raise ValueError(
            'Argument `etas` must contain at least one non-zero value, otherwise '
            'consider using `mesolve`.'
        )
    if torch.any(etas < 0.0) or torch.any(etas > 1.0):
        raise ValueError('Argument `etas` must contain values between 0 and 1.')

    # split jump operators between purely dissipative (eta = 0) and monitored (eta != 0)
    mask = etas == 0.0
    meas_ops, etas = jump_ops[~mask], etas[~mask]

    # convert t_meas to a tensor
    t_meas = torch.as_tensor(
        [] if t_meas is None else t_meas, dtype=options.rdtype, device=options.device
    )
    check_time_tensor(t_meas, arg_name='t_meas', allow_empty=True)

    # define random number generator from seed
    generator = torch.Generator(device=options.device)
    generator.seed() if seed is None else generator.manual_seed(seed)

    # define the solver
    args = (H_batched, rho0_batched, t_save, exp_ops, options)
    kwargs = dict(
        jump_ops=jump_ops,
        meas_ops=meas_ops,
        etas=etas,
        generator=generator,
        t_meas=t_meas,
    )
    if isinstance(options, Euler):
        solver = SMEEuler(*args, **kwargs)
    else:
        raise ValueError(f'Solver options {obj_type_str(options)} is not supported.')

    # compute the result
    solver.run()

    # get saved tensors and restore correct batching
    result = solver.result
    result.y_save = formatter.unbatch(result.y_save)
    result.exp_save = formatter.unbatch(result.exp_save)
    result.meas_save = formatter.unbatch(result.meas_save)

    return result
