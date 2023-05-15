from torch import Tensor

from ..solvers.ode.fixed_solver import FixedSolver
from .me_solver import MESolver


class MEEuler(MESolver, FixedSolver):
    def forward(self, t: float, dt: float, rho: Tensor) -> Tensor:
        # rho: (b_H, b_rho, n, n) -> (b_H, b_rho, n, n)
        return rho + dt * self.lindbladian(t, rho)
