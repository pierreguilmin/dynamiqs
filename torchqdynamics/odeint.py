import warnings

import torch

from .solver import AdaptativeStep, FixedStep


def odeint(qsolver, y0, tsave, sensitivity='autograd', variables=None):
    # check arguments
    tsave = check_tsave(tsave)
    if (variables is not None) and (sensitivity in [None, 'autograd']):
        warnings.warn('Argument `variables` was supplied in `odeint` but not used.')

    # dispatch to appropriate odeint subroutine
    if sensitivity is None:
        return odeint_inplace(qsolver, y0, tsave)
    elif sensitivity == 'autograd':
        return odeint_main(qsolver, y0, tsave)
    elif sensitivity == 'adjoint':
        return odeint_adjoint(qsolver, y0, tsave, variables)


def odeint_main(qsolver, y0, tsave):
    if isinstance(qsolver.options, FixedStep):
        return _fixed_odeint(qsolver, y0, tsave)
    elif isinstance(qsolver.options, AdaptativeStep):
        return _adaptive_odeint(qsolver, y0, tsave)


def odeint_inplace(qsolver, y0, tsave):
    # TODO: Simple solution for now so torch does not store gradients. This
    #       is probably slower than a genuine in-place solver.
    with torch.no_grad():
        return odeint_main(qsolver, y0, tsave)


def odeint_adjoint(qsolver, y0, tsave, variables):
    raise NotImplementedError


def _adaptive_odeint(qsolver, y0, tsave):
    raise NotImplementedError


def _fixed_odeint(qsolver, y0, tsave):
    # initialize save tensor
    ysave = torch.zeros((len(tsave), ) + y0.shape).to(y0)
    save_counter = 0
    if tsave[0] == 0:
        ysave[0] = y0
        save_counter += 1

    # get qsolver fixed time step
    dt = qsolver.options.dt

    # run the ODE routine
    t, y = 0, y0
    while t < tsave[-1]:
        # check if final time is reached
        if t + dt > tsave[-1]:
            dt = tsave[-1] - t

        # iterate solution
        y = qsolver.forward(t, dt, y)
        t = t + dt

        # save solution
        if t >= tsave[save_counter]:
            ysave[save_counter] = y
            save_counter += 1

    return tsave, ysave


def check_tsave(tsave):
    """Check tsave is a sorted 1-D `torch.tensor`."""
    if isinstance(tsave, (list, np.ndarray)):
        tsave = torch.cat(tsave)
    if tsave.dim != 1 or len(tsave) == 0:
        raise ValueError('Argument `tsave` should be a non-empty 1-D torch.Tensor.')
    if not torch.all(torch.diff(tsave) > 0):
        raise ValueError(
            'Argument `tsave` is not sorted in ascending order '
            'or contains duplicate values.'
        )
    return tsave
