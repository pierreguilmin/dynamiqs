from __future__ import annotations

import warnings
from abc import abstractmethod
from functools import partial

import diffrax as dx
import equinox as eqx
from jax import Array
from jaxtyping import PyTree, Scalar

from ...gradient import Autograd, CheckpointAutograd
from ...result import Result
from .abstract_integrator import BaseIntegrator
from .interfaces import AbstractTimeInterface, MEInterface, SEInterface, SolveInterface
from .save_mixin import AbstractSaveMixin, PropagatorSaveMixin, SolveSaveMixin


class FixedStepInfos(eqx.Module):
    nsteps: Array

    def __str__(self) -> str:
        if self.nsteps.ndim >= 1:
            # note: fixed step methods always make the same number of steps
            return f'{int(self.nsteps.mean())} steps | infos shape {self.nsteps.shape}'
        return f'{self.nsteps} steps'


class AdaptiveStepInfos(eqx.Module):
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


class DiffraxIntegrator(BaseIntegrator, AbstractSaveMixin, AbstractTimeInterface):
    """Integrator using the Diffrax library."""

    diffrax_solver: dx.AbstractSolver
    fixed_step: bool

    @property
    def stepsize_controller(self) -> dx.AbstractStepSizeController:
        if self.fixed_step:
            return dx.ConstantStepSize()
        else:
            return dx.PIDController(
                rtol=self.method.rtol,
                atol=self.method.atol,
                safety=self.method.safety_factor,
                factormin=self.method.min_factor,
                factormax=self.method.max_factor,
                jump_ts=self.discontinuity_ts,
            )

    @property
    def dt0(self) -> float | None:
        return self.method.dt if self.fixed_step else None

    @property
    def max_steps(self) -> int:
        # TODO: fix hard-coded max_steps for fixed methods
        return 100_000 if self.fixed_step else self.method.max_steps

    @property
    @abstractmethod
    def terms(self) -> dx.AbstractTerm:
        pass

    @property
    def adjoint(self) -> dx.AbstractAdjoint:
        if self.gradient is None:
            return dx.RecursiveCheckpointAdjoint()
        elif isinstance(self.gradient, CheckpointAutograd):
            return dx.RecursiveCheckpointAdjoint(self.gradient.ncheckpoints)
        elif isinstance(self.gradient, Autograd):
            return dx.DirectAdjoint()
        else:
            raise TypeError(f'Unknown gradient type {self.gradient}')

    def diffeqsolve(
        self,
        t0: Scalar,
        t1: Scalar,
        y0: PyTree,
        saveat: dx.SaveAt,
        event: dx.Event | None = None,
    ) -> dx.Solution:
        with warnings.catch_warnings():
            # TODO: remove once complex support is stabilized in diffrax
            warnings.simplefilter('ignore', UserWarning)

            # === solve differential equation with diffrax
            return dx.diffeqsolve(
                self.terms,
                self.diffrax_solver,
                t0=t0,
                t1=t1,
                dt0=self.dt0,
                y0=y0,
                saveat=saveat,
                stepsize_controller=self.stepsize_controller,
                adjoint=self.adjoint,
                event=event,
                max_steps=self.max_steps,
                progress_meter=self.options.progress_meter.to_diffrax(),
            )

    def run(self) -> Result:
        # === prepare diffrax arguments
        fn = lambda t, y, args: self.save(y)  # noqa: ARG005
        subsaveat_a = dx.SubSaveAt(ts=self.ts, fn=fn)  # save solution regularly
        subsaveat_b = dx.SubSaveAt(t1=True)  # save last state
        saveat = dx.SaveAt(subs=[subsaveat_a, subsaveat_b])

        # === solve differential equation
        solution = self.diffeqsolve(t0=self.t0, t1=self.t1, y0=self.y0, saveat=saveat)

        # === collect and return results
        saved = self.postprocess_saved(*solution.ys)
        return self.result(saved, infos=self.infos(solution.stats))

    def infos(self, stats: dict[str, Array]) -> PyTree:
        if self.fixed_step:
            return FixedStepInfos(stats['num_steps'])
        else:
            return AdaptiveStepInfos(
                stats['num_steps'],
                stats['num_accepted_steps'],
                stats['num_rejected_steps'],
            )


class SEDiffraxIntegrator(DiffraxIntegrator, SEInterface):
    """Integrator solving the Schrödinger equation with Diffrax."""

    @property
    def terms(self) -> dx.AbstractTerm:
        # define Schrödinger term d|psi>/dt = - i H |psi>
        vector_field = lambda t, y, _: -1j * self.H(t) @ y
        return dx.ODETerm(vector_field)


class SEPropagatorDiffraxIntegrator(SEDiffraxIntegrator, PropagatorSaveMixin):
    """Integrator computing the propagator of the Schrödinger equation using the Diffrax
    library.
    """


sepropagator_euler_integrator_constructor = partial(
    SEPropagatorDiffraxIntegrator, diffrax_solver=dx.Euler(), fixed_step=True
)
sepropagator_dopri5_integrator_constructor = partial(
    SEPropagatorDiffraxIntegrator, diffrax_solver=dx.Dopri5(), fixed_step=False
)
sepropagator_dopri8_integrator_constructor = partial(
    SEPropagatorDiffraxIntegrator, diffrax_solver=dx.Dopri8(), fixed_step=False
)
sepropagator_tsit5_integrator_constructor = partial(
    SEPropagatorDiffraxIntegrator, diffrax_solver=dx.Tsit5(), fixed_step=False
)
sepropagator_kvaerno3_integrator_constructor = partial(
    SEPropagatorDiffraxIntegrator, diffrax_solver=dx.Kvaerno3(), fixed_step=False
)
sepropagator_kvaerno5_integrator_constructor = partial(
    SEPropagatorDiffraxIntegrator, diffrax_solver=dx.Kvaerno5(), fixed_step=False
)


class SESolveDiffraxIntegrator(SEDiffraxIntegrator, SolveSaveMixin, SolveInterface):
    """Integrator computing the time evolution of the Schrödinger equation using the
    Diffrax library.
    """


sesolve_euler_integrator_constructor = partial(
    SESolveDiffraxIntegrator, diffrax_solver=dx.Euler(), fixed_step=True
)
sesolve_dopri5_integrator_constructor = partial(
    SESolveDiffraxIntegrator, diffrax_solver=dx.Dopri5(), fixed_step=False
)
sesolve_dopri8_integrator_constructor = partial(
    SESolveDiffraxIntegrator, diffrax_solver=dx.Dopri8(), fixed_step=False
)
sesolve_tsit5_integrator_constructor = partial(
    SESolveDiffraxIntegrator, diffrax_solver=dx.Tsit5(), fixed_step=False
)
sesolve_kvaerno3_integrator_constructor = partial(
    SESolveDiffraxIntegrator, diffrax_solver=dx.Kvaerno3(), fixed_step=False
)
sesolve_kvaerno5_integrator_constructor = partial(
    SESolveDiffraxIntegrator, diffrax_solver=dx.Kvaerno5(), fixed_step=False
)


class MEDiffraxIntegrator(DiffraxIntegrator, MEInterface):
    """Integrator solving the Lindblad master equation with Diffrax."""

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
            L, H = self.L(t), self.H(t)
            Hnh = -1j * H + sum([-0.5 * _L.dag() @ _L for _L in L])
            tmp = Hnh @ y + sum([0.5 * _L @ y @ _L.dag() for _L in L])
            return tmp + tmp.dag()

        return dx.ODETerm(vector_field)


class MESolveDiffraxIntegrator(MEDiffraxIntegrator, SolveSaveMixin, SolveInterface):
    """Integrator computing the time evolution of the Lindblad master equation using the
    Diffrax library.
    """


mesolve_euler_integrator_constructor = partial(
    MESolveDiffraxIntegrator, diffrax_solver=dx.Euler(), fixed_step=True
)
mesolve_dopri5_integrator_constructor = partial(
    MESolveDiffraxIntegrator, diffrax_solver=dx.Dopri5(), fixed_step=False
)
mesolve_dopri8_integrator_constructor = partial(
    MESolveDiffraxIntegrator, diffrax_solver=dx.Dopri8(), fixed_step=False
)
mesolve_tsit5_integrator_constructor = partial(
    MESolveDiffraxIntegrator, diffrax_solver=dx.Tsit5(), fixed_step=False
)
mesolve_kvaerno3_integrator_constructor = partial(
    MESolveDiffraxIntegrator, diffrax_solver=dx.Kvaerno3(), fixed_step=False
)
mesolve_kvaerno5_integrator_constructor = partial(
    MESolveDiffraxIntegrator, diffrax_solver=dx.Kvaerno5(), fixed_step=False
)
