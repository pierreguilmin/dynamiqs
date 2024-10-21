from __future__ import annotations

import warnings
from abc import abstractmethod

import diffrax as dx
import equinox as eqx
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from ...gradient import Autograd, CheckpointAutograd
from ...result import Result, DSMESolveSaved, Saved
from ...utils.quantum_utils.general import dag, trace
from .abstract_integrator import BaseIntegrator, DSMEBaseIntegrator
from .save_mixin import SaveMixin
from .interfaces import SEInterface, MEInterface, DSMEInterface


class DiffraxIntegrator(BaseIntegrator, SaveMixin):
    """Integrator using the Diffrax library."""

    # subclasses should implement: stepsize_controller, dt0, max_steps, diffrax_solver,
    # terms, discontinuity_ts, infos()

    @property
    @abstractmethod
    def stepsize_controller(self) -> dx.AbstractStepSizeController:
        pass

    @property
    @abstractmethod
    def dt0(self) -> float | None:
        pass

    @property
    @abstractmethod
    def max_steps(self) -> int:
        pass

    @property
    @abstractmethod
    def diffrax_solver(self) -> dx.AbstractSolver:
        pass

    @property
    @abstractmethod
    def terms(self) -> dx.AbstractTerm:
        pass

    @property
    def saveat(self) -> dx.SaveAt:
        fn = lambda t, y, args: self.save(y)  # noqa: ARG005
        subsaveat_a = dx.SubSaveAt(ts=self.ts, fn=fn)  # save solution regularly
        subsaveat_b = dx.SubSaveAt(t1=True)  # save last state
        return dx.SaveAt(subs=[subsaveat_a, subsaveat_b])

    def solution_to_saved(self, solution: dx.Solution) -> Saved:
        return solution.ys[0]

    def run(self) -> Result:
        with warnings.catch_warnings():
            # TODO: remove once complex support is stabilized in diffrax
            warnings.simplefilter('ignore', UserWarning)
            # TODO: remove once https://github.com/patrick-kidger/diffrax/issues/445 is
            # closed
            warnings.simplefilter('ignore', FutureWarning)

            # === prepare adjoint argument
            if self.gradient is None:
                adjoint = dx.RecursiveCheckpointAdjoint()
            elif isinstance(self.gradient, CheckpointAutograd):
                adjoint = dx.RecursiveCheckpointAdjoint(self.gradient.ncheckpoints)
            elif isinstance(self.gradient, Autograd):
                adjoint = dx.DirectAdjoint()

            # === solve differential equation with diffrax
            solution = dx.diffeqsolve(
                self.terms,
                self.diffrax_solver,
                t0=self.t0,
                t1=self.t1,
                dt0=self.dt0,
                y0=self.y0,
                saveat=self.saveat,
                stepsize_controller=self.stepsize_controller,
                adjoint=adjoint,
                max_steps=self.max_steps,
                progress_meter=self.options.progress_meter.to_diffrax(),
            )

        # === collect and return results
        saved = self.postprocess_saved(self.solution_to_saved(solution), solution.ys[1])
        return self.result(saved, infos=self.infos(solution.stats))

    @abstractmethod
    def infos(self, stats: dict[str, Array]) -> PyTree:
        pass


class FixedStepDiffraxIntegrator(DiffraxIntegrator):
    """Integrator using a fixed step Diffrax solver."""

    # subclasses should implement: diffrax_solver, terms, discontinuity_ts

    class Infos(eqx.Module):
        nsteps: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                # note: fixed step solvers always make the same number of steps
                return (
                    f'{int(self.nsteps.mean())} steps | infos shape {self.nsteps.shape}'
                )
            return f'{self.nsteps} steps'

    def infos(self, stats: dict[str, Array]) -> PyTree:
        return self.Infos(stats['num_steps'])

    @property
    def stepsize_controller(self) -> dx.AbstractStepSizeController:
        return dx.ConstantStepSize()

    @property
    def dt0(self) -> float | None:
        return self.solver.dt

    @property
    def max_steps(self) -> int:
        return 100_000  # TODO: fix hard-coded max_steps


class AdaptiveStepDiffraxIntegrator(DiffraxIntegrator):
    """Integrator using an adaptive step Diffrax solver."""

    # subclasses should implement: diffrax_solver, terms, discontinuity_ts

    class Infos(eqx.Module):
        nsteps: Array
        naccepted: Array
        nrejected: Array

        def __str__(self) -> str:
            if self.nsteps.ndim >= 1:
                return (
                    f'avg. {self.nsteps.mean():.1f} steps ({self.naccepted.mean():.1f}'
                    f' accepted, {self.nrejected.mean():.1f} rejected) | infos shape'
                    f' {self.nsteps.shape}'
                )
            return (
                f'{self.nsteps} steps ({self.naccepted} accepted,'
                f' {self.nrejected} rejected)'
            )

    def infos(self, stats: dict[str, Array]) -> PyTree:
        return self.Infos(
            stats['num_steps'], stats['num_accepted_steps'], stats['num_rejected_steps']
        )

    @property
    def stepsize_controller(self) -> dx.AbstractStepSizeController:
        return dx.PIDController(
            rtol=self.solver.rtol,
            atol=self.solver.atol,
            safety=self.solver.safety_factor,
            factormin=self.solver.min_factor,
            factormax=self.solver.max_factor,
            jump_ts=self.discontinuity_ts,
        )

    @property
    def dt0(self) -> float | None:
        return None

    @property
    def max_steps(self) -> int:
        return self.solver.max_steps


# fmt: off
# ruff: noqa
class EulerIntegrator(FixedStepDiffraxIntegrator): diffrax_solver = dx.Euler()
class Dopri5Integrator(AdaptiveStepDiffraxIntegrator): diffrax_solver = dx.Dopri5()
class Dopri8Integrator(AdaptiveStepDiffraxIntegrator): diffrax_solver = dx.Dopri8()
class Tsit5Integrator(AdaptiveStepDiffraxIntegrator): diffrax_solver = dx.Tsit5()
class Kvaerno3Integrator(AdaptiveStepDiffraxIntegrator): diffrax_solver = dx.Kvaerno3()
class Kvaerno5Integrator(AdaptiveStepDiffraxIntegrator): diffrax_solver = dx.Kvaerno5()
class MilsteinIntegrator(AdaptiveStepDiffraxIntegrator): diffrax_solver = dx.HalfSolver(dx.ItoMilstein())
# fmt: on


class SEDiffraxIntegrator(DiffraxIntegrator, SEInterface):
    """Integrator solving the Schrödinger equation with Diffrax."""

    # subclasses should implement: diffrax_solver, discontinuity_ts

    @property
    def terms(self) -> dx.AbstractTerm:
        # define Schrödinger term d|psi>/dt = - i H |psi>
        vector_field = lambda t, y, _: -1j * self.H(t) @ y
        return dx.ODETerm(vector_field)


class MEDiffraxIntegrator(DiffraxIntegrator, MEInterface):
    """Integrator solving the Lindblad master equation with Diffrax."""

    # subclasses should implement: diffrax_solver, discontinuity_ts

    @property
    def terms(self) -> dx.AbstractTerm:
        # define Lindblad term drho/dt

        # The Lindblad equation for a single loss channel is:
        # (1) drho/dt = -i [H, rho] + L @ rho @ Ld - 0.5 Ld @ L @ rho - 0.5 rho @ Ld @ L
        # An alternative but similar equation is:
        # (2) drho/dt = (-i H @ rho + 0.5 L @ rho @ Ld - 0.5 Ld @ L @ rho) + h.c.
        # While (1) and (2) are equivalent assuming that rho is hermitian, they differ
        # once you take into account numerical errors.
        # Decomposing rho = rho_s + rho_a with Hermitian rho_s and anti-Hermitian rho_a,
        # we get that:
        #  - if rho evolves according to (1), both rho_s and rho_a also evolve
        #    according to (1);
        #  - if rho evolves according to (2), rho_s evolves closely to (1) up
        #    to a constant error that depends on rho_a (which is small up to numerical
        #    precision), while rho_a is strictly constant.
        # In practice, we still use (2) because it involves less matrix multiplications,
        # and is thus more efficient numerically with only a negligible numerical error
        # induced on the dynamics.

        def vector_field(t, y, _):  # noqa: ANN001, ANN202
            Ls = jnp.stack([L(t) for L in self.Ls])
            Lsd = dag(Ls)
            LdL = (Lsd @ Ls).sum(0)
            tmp = (-1j * self.H(t) - 0.5 * LdL) @ y + 0.5 * (Ls @ y @ Lsd).sum(0)
            return tmp + dag(tmp)

        return dx.ODETerm(vector_field)


# state for the diffrax solver for diffusive SMEs
class YDSME(eqx.Module):
    # The SME state at each time is (rho, Y) with rho the state (integrated from initial
    # to current time) and Y the signal (integrated from initial to current time).
    rho: Array
    Y: Array


class DSMEDiffraxIntegrator(DiffraxIntegrator, DSMEBaseIntegrator, DSMEInterface):
    """Integrator solving the diffusive SME with Diffrax."""

    # subclasses should implement: diffrax_solver, discontinuity_ts

    @property
    def saveat(self) -> dx.SaveAt:
        saveat = super().saveat

        # === define save function to save measurement results at each time in tmeas
        fn = lambda t, y, args: y.Y  # noqa: ARG005
        save_c = dx.SubSaveAt(ts=self.tmeas, fn=fn)  # save measurement results
        saveat.subs.append(save_c)

        return saveat

    @property
    def wiener(self) -> dx.VirtualBrownianTree:
        # === define wiener process
        return dx.VirtualBrownianTree(
            self.t0, self.t1, tol=1e-3, shape=(len(self.etas),), key=self.key
        )  # todo: fix hard-coded tol

    @property
    def terms(self) -> dx.AbstractTerm:
        # The diffusive SME is a coupled system of SDEs with state (rho, Y), which for
        # a single detector writes:
        #   drho = Lcal(rho)     dt + (Ccal(rho) - Tr[Ccal(rho)] rho) dWt
        #   dY   = Tr[Ccal(rho)] dt + dWt
        # with
        # - Lcal the Liouvillian
        # - Ccal the superoperator defined by Ccal(rho) = sqrt(eta) (L @ rho + rho @ Ld)

        # To follow Diffrax-way, we write
        # (1) The deterministic part (everything before dt) in
        #     `vector_field_deterministic`: (rho, Y) -> (Lcal(rho), Tr[Ccal(rho)])
        # (2) The stochastic part (everything before dWt) in
        #     `vector_field_stochastic`: (rho, Y) -> (Ccal(rho) - Tr[Ccal(rho)] rho, 1)
        # (3) The measurement part (how to mix the stochastic part and the dWt) in
        #     `measurement_term`, which is just a simple multiplication of the
        #     stochastic part (2) with dWt.

        def Ccal(t, rho):
            Lms = jnp.stack([L(t) for L in self.Lms])  # (nLm, n, n)
            Lms_rho = Lms @ rho
            etas = self.etas[:, None, None]  # (nLm, 1, 1)
            return jnp.sqrt(etas) * (Lms_rho + dag(Lms_rho))  # (nLm, n, n)

        # === define deterministic term
        def vector_field_deterministic(t, y, _):  # noqa: ANN001, ANN202
            # state: Lcal(rho) (see MEDiffraxIntegrator)
            Ls = jnp.stack([L(t) for L in self.Ls])
            Lsd = dag(Ls)
            LdL = (Lsd @ Ls).sum(0)
            H = self.H(t)
            tmp = (-1j * H - 0.5 * LdL) @ y.rho + 0.5 * (Ls @ y.rho @ Lsd).sum(0)
            drho = tmp + dag(tmp)

            # signal: Tr[Ccal(rho)]
            dYt = trace(Ccal(t, y.rho)).real  # (nLm,)

            return YDSME(drho, dYt)

        deterministic_term = dx.ODETerm(vector_field_deterministic)

        # === define stochastic term
        def vector_field_stochastic(t, y, _):  # noqa: ANN001, ANN202
            # state: Ccal(rho) - Tr[Ccal(rho)] rho
            Ccal_rho = Ccal(t, y.rho)
            drho = Ccal_rho - trace(Ccal_rho).real[:, None, None] * y.rho  # (nLm, n, n)

            # signal: 1
            dYt = 1

            return YDSME(drho, dYt)

        control = self.wiener

        # === define measurement term
        class MeasurementTerm(dx.ControlTerm):
            def prod(self, vf: dx.VF, control: dx.Control) -> dx.Y:
                dW = control
                drho = (vf.rho * dW[:, None, None]).sum(0)  # (n, n)
                return YDSME(drho, dW)

        measurement_term = MeasurementTerm(vector_field_stochastic, control)

        # === combine and return both terms
        return dx.MultiTerm(deterministic_term, measurement_term)

    def solution_to_saved(self, solution: dx.Solution) -> Saved:
        saved = solution.ys[0]
        Ysave = solution.ys[2]
        return DSMESolveSaved(saved.ysave, saved.extra, saved.Esave, Ysave)
