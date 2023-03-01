from dataclasses import dataclass

import torch

from .odeint import odeint

# --------------------------------------------------------------------------------------
#     Main mesolve function
# --------------------------------------------------------------------------------------


def mesolve(
    H, jump_ops, rho0, tsave, solver=Rouchon(), sensitivity='autograd',
    model_params=None
):
    # Define the QSolver
    if isinstance(solver, Rouchon):
        if solver.order == 1:
            qsolver = MERouchon1(H, jump_ops, solver)
        elif solver.order == 2:
            qsolver = MERouchon2(H, jump_ops, solver)
    else:
        raise NotImplementedError

    # Compute the result
    return odeint(
        qsolver, rho0, tsave, sensitivity=sensitivity, model_params=model_params
    )


# --------------------------------------------------------------------------------------
#     ME QSolver classes
# --------------------------------------------------------------------------------------


class QSolver:
    def __init__(self):
        pass


class MERouchon(QSolver):
    def __init__(self, H, jump_ops, solver_options):
        self.H = H
        self.jump_ops = jump_ops
        self.jumpdag_ops = jump_ops.adjoint()
        self.sum_nojump = (self.jumpdag_ops @ self.jump_ops).sum(dim=0)
        self.I = torch.eye(H(0).shape[-1]).to(H(0))
        self.options = solver_options


class MERouchon1(MERouchon):
    def forward(self, t, dt, rho):
        """Compute rho(t+dt) using a Rouchon method of order 1"""
        # Non-hermitian Hamiltonian at time t
        H_nh = self.H(t) - 0.5j * self.sum_nojump

        # Build time-dependent Kraus operators
        M0 = self.I - 1j * dt * H_nh
        M1s = self.jump_ops

        # Compute rho(t+dt)
        drho = M0 @ rho @ M0.adjoint(
        ) + dt * (self.jump_ops @ rho.unsqueeze(0) @ self.jumpdag_ops).sum(dim=0)

        return drho

    def forward_adjoint(self, t, dt, phi):
        raise NotImplementedError


class MERouchon2(MERouchon):
    def forward(self, t, dt, rho):
        """Compute rho(t+dt) using a Rouchon method of order 2"""
        # Non-hermitian Hamiltonian at time t
        H_nh = self.H(t) - 0.5j * self.sum_nojump

        # Build time-dependent Kraus operators
        # TODO: Add the missing time derivative term in -0.5j * dt**2 * \dot{H}
        M0 = self.I - 1j * dt * H_nh - 0.5 * dt**2 * H_nh @ H_nh
        M1s = 0.5 * (self.Ls @ M0 + M0 @ self.Ls)

        # Compute rho(t+dt)
        drho_ = dt * torch.sum(M1s @ rho.unsqueeze(0) @ M1s.adjoint(), dim=0)
        drho = 0.5 * dt * torch.sum(M1s @ drho_.unsqueeze(0) @ M1s.adjoint(), dim=0)
        drho += drho_ + M0 @ rho @ M0.adjoint()

        return drho

    def forward_adjoint(self, t, dt, phi):
        raise NotImplementedError


# --------------------------------------------------------------------------------------
#     ME Solver Options
# --------------------------------------------------------------------------------------

# See the PR by @abocquet at https://github.com/PierreGuilmin/torchqdynamics/pull/10


@dataclass
class Rouchon:
    dt: float = 1e-2
    order: float = 1
    stepclass: str = 'fixed'
