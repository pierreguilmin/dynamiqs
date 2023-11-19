from __future__ import annotations

from torch import Tensor

from ..solvers.ode.fixed_solver import AdjointFixedSolver
from .me_solver import MESolver


class MEEuler(MESolver, AdjointFixedSolver):
    def forward(self, t: float, rho: Tensor) -> Tensor:
        # rho: (b_H, b_L, b_rho, n, n) -> (b_H, b_L, b_rho, n, n)
        return rho + self.dt * self.lindbladian(t, rho)

    def backward_augmented(
        self, t: float, rho: Tensor, phi: Tensor
    ) -> tuple[Tensor, Tensor]:
        # rho: (b_H, b_L, b_rho, n, n) -> (b_H, b_L, b_rho, n, n)
        # phi: (b_H, b_L, b_rho, n, n) -> (b_H, b_L, b_rho, n, n)
        rho = rho - self.dt * self.lindbladian(t, rho)
        phi = phi + self.dt * self.adjoint_lindbladian(t, phi)
        return rho, phi
