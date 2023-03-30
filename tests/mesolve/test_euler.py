import pytest

import torchqdynamics as tq

from .mesolver_test import MESolverTest
from .open_system import leaky_cavity_8


class TestMEEuler(MESolverTest):
    def test_batching(self):
        solver = tq.solver.Euler(dt=1e-2)
        self._test_batching(solver, leaky_cavity_8)

    @pytest.mark.skip(reason='failing - to fix')
    @pytest.mark.long
    def test_rho_save(self):
        solver = tq.solver.Euler(dt=1e-4)
        self._test_rho_save(solver, leaky_cavity_8, nt=51, rtol=1e-3, atol=1e-3)
