from __future__ import annotations

import warnings
from abc import abstractmethod
from functools import partial

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.internal import while_loop
from jax import Array
from jaxtyping import PyTree, Scalar

from ...gradient import Autograd, CheckpointAutograd
from ...qarrays.qarray import QArray
from ...qarrays.utils import stack
from ...result import MCJumpResult, MCNoJumpResult, Result, Saved
from ...utils.general import dag, expect, norm, unit
from .abstract_integrator import BaseIntegrator
from .interfaces import (
    AbstractTimeInterface,
    MCInterface,
    MEInterface,
    SEInterface,
    SolveInterface,
)
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


class SaveState(eqx.Module):
    ts: Array
    saved: PyTree
    save_index: int
    jump_times: Array


class State(eqx.Module):
    save_state: PyTree[SaveState]
    final_state: Array
    final_time: float
    num_jumps: int
    key: Array


def _inner_buffers(save_state: SaveState) -> tuple[Array, Saved, Array]:
    assert type(save_state) is SaveState
    return save_state.ts, save_state.saved, save_state.jump_times


def _outer_buffers(state: State) -> tuple[Array, Saved, Array]:
    assert type(state) is State
    save_state = state.save_state
    return save_state.ts, save_state.saved, save_state.jump_times


class MCSolveDiffraxIntegrator(
    DiffraxIntegrator, MCInterface, SolveSaveMixin, SolveInterface
):
    """Integrator computing the time evolution of the Monte-Carlo unraveling of the
    Lindblad master equation using the Diffrax library.
    """

    def jump_result(
        self,
        saved: Saved,
        jump_times: Array,
        num_jumps: Array,
        infos: PyTree | None = None,
    ) -> Result:
        return MCJumpResult(
            self.ts,
            self.method,
            self.gradient,
            self.options,
            saved,
            infos,
            jump_times,
            num_jumps,
        )

    def no_jump_result(
        self, saved: Saved, no_jump_prob: Array, infos: PyTree | None = None
    ) -> Result:
        return MCNoJumpResult(
            self.ts,
            self.method,
            self.gradient,
            self.options,
            saved,
            infos,
            no_jump_prob,
        )

    def result(
        self,
        saved: Saved,
        no_jump_result: Result,
        jump_result: Result,
        infos: PyTree | None = None,
    ) -> Result:
        return self.result_class(
            self.ts,
            self.method,
            self.gradient,
            self.options,
            saved,
            infos,
            no_jump_result,
            jump_result,
        )

    @property
    def terms(self) -> dx.AbstractTerm:
        def vector_field(t, y, _):  # noqa: ANN001, ANN202
            L, H = self.L(t), self.H(t)
            return -1j * H @ y + sum([-0.5 * _L.dag() @ (_L @ y) for _L in L])

        return dx.ODETerm(vector_field)

    def run(self) -> Result:
        # === run no jump, extract no-jump probability
        no_jump_solution = self._solve_until_jump(self.y0, self.t0, self.ts, 0.0)
        no_jump_saved = jax.vmap(self.save)(unit(no_jump_solution.ys[0]))
        no_jump_prob = norm(no_jump_solution.ys[1][0]) ** 2

        # === run jump trajectories
        jump_state_fun = jax.vmap(self._loop_over_jumps, in_axes=(0, None))
        jump_state = jump_state_fun(self.keys, no_jump_prob)
        jump_times = jump_state.save_state.jump_times
        num_jumps = jump_state.num_jumps

        # === save and postprocess jump and no-jump results
        no_jump_saved = self.postprocess_saved(
            no_jump_saved, unit(no_jump_solution.ys[1])
        )
        no_jump_result = self.no_jump_result(
            no_jump_saved, no_jump_prob, infos=self.infos(no_jump_solution.stats)
        )
        jump_saved = self.postprocess_saved(
            jump_state.save_state.saved, jump_state.final_state[None]
        )
        # TODO save stats for jumps?
        jump_result = self.jump_result(jump_saved, jump_times, num_jumps)

        rho = self._average_jump_and_no_jump(
            no_jump_result.states, jump_result.states, no_jump_prob
        )
        final_rho = self._average_jump_and_no_jump(
            no_jump_result.final_state, jump_result.final_state, no_jump_prob
        )
        saved = jax.vmap(self.save)(rho)
        saved = self.postprocess_saved(saved, final_rho)

        return self.result(saved, no_jump_result, jump_result)

    def _solve_until_jump(
        self, y0: QArray, t0: Array, tsave: Array, rand: Array | float
    ) -> dx.Solution:
        # === prepare saveat
        subsaveat_a = dx.SubSaveAt(ts=tsave)  # save solution regularly
        subsaveat_b = dx.SubSaveAt(t1=True)  # save last state
        saveat = dx.SaveAt(subs=[subsaveat_a, subsaveat_b])

        # === prepare event
        def cond_fn(t: Scalar, y: QArray, *args, **kwargs) -> Array:
            return norm(y) ** 2 - rand

        event = dx.Event(cond_fn, self.root_finder)

        # === solve differential equation with diffrax
        return self.diffeqsolve(t0=t0, t1=self.t1, y0=y0, saveat=saveat, event=event)

    def _loop_over_jumps(self, key: Array, no_jump_prob: Array) -> State:
        # loop over jumps until the simulation reaches the final time

        rand_key, sample_key = jax.random.split(key)
        rand = jax.random.uniform(rand_key, minval=no_jump_prob)
        first_result = self._solve_until_jump(self.y0, self.t0, self.ts, rand)
        first_jump_time = first_result.ts[1][0]
        time_diffs = self.ts - first_jump_time
        # want to start with a time after the jump
        time_diffs = jnp.where(time_diffs < 0, jnp.inf, time_diffs)
        save_index = jnp.argmin(time_diffs)
        # allocate memory for the state that is updated during the loop
        initial_save_state = SaveState(
            ts=first_result.ts[0],
            saved=jax.vmap(self.save)(unit(first_result.ys[0])),
            save_index=save_index,
            jump_times=jnp.full(self.options.max_jumps, jnp.nan),
        )
        initial_state = State(
            save_state=initial_save_state,
            final_state=first_result.ys[1][0, :, :],
            final_time=first_jump_time,
            num_jumps=0,
            key=sample_key,
        )

        def _outer_cond_fun(state: State) -> bool:
            return state.final_time < self.t1

        def _outer_body_fun(state: State) -> State:
            jump_time = state.final_time
            psi = state.final_state
            _next_key, _rand_key, _sample_key = jax.random.split(state.key, num=3)
            jump_op = self._sample_jump_ops(jump_time, psi, _sample_key)
            psi_after_jump = unit(jump_op @ psi)
            # new_times needs to have a static size: fill it with times from self.ts
            # and nans. The saved states at times from the original self.ts are saved
            # below, while the states at the nan times are ignored.
            new_times = jnp.sort(jnp.where(self.ts > jump_time, self.ts, jnp.nan))
            # This new random value should be uniform over [0, 1), not restricted to
            # [no_jump_prob, 1) like in the first case.
            _rand = jax.random.uniform(_rand_key)
            result_after_jump = self._solve_until_jump(
                psi_after_jump, jump_time, new_times, _rand
            )
            final_time = result_after_jump.ts[1][0]

            def save_ts_and_states(save_state: SaveState) -> SaveState:
                # Save intermediate states and expectation values. Based on the code
                # in diffrax/_integrate.py which can be found at:
                # https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_integrate.py#L427-L458  # noqa E501
                def _inner_cond_fun(_save_state: SaveState) -> bool:
                    return (self.ts[_save_state.save_index] <= final_time) & (
                        _save_state.save_index < len(self.ts)
                    )

                def _inner_body_fun(_save_state: SaveState) -> SaveState:
                    _t = self.ts[_save_state.save_index]
                    t_idx_in_new_times = jnp.nanargmin(jnp.abs(new_times - _t))
                    _y = result_after_jump.ys[0][t_idx_in_new_times]
                    _ts = _save_state.ts.at[_save_state.save_index].set(_t)
                    _saved = jax.tree.map(
                        lambda __y, __saved: __saved.at[_save_state.save_index].set(
                            __y
                        ),
                        self.save(unit(_y)),
                        _save_state.saved,
                    )
                    return SaveState(
                        ts=_ts,
                        saved=_saved,
                        save_index=_save_state.save_index + 1,
                        jump_times=_save_state.jump_times,
                    )

                return while_loop(
                    _inner_cond_fun,
                    _inner_body_fun,
                    save_state,
                    max_steps=len(self.ts),
                    buffers=_inner_buffers,
                    kind='checkpointed',
                )

            save_state = save_ts_and_states(state.save_state)
            jump_times = save_state.jump_times.at[state.num_jumps].set(jump_time)
            save_state = SaveState(
                ts=save_state.ts,
                saved=save_state.saved,
                save_index=save_state.save_index,
                jump_times=jump_times,
            )
            return State(
                save_state=save_state,
                final_state=result_after_jump.ys[1][0],
                final_time=final_time,
                num_jumps=state.num_jumps + 1,
                key=_next_key,
            )

        return while_loop(
            _outer_cond_fun,
            _outer_body_fun,
            initial_state,
            max_steps=self.options.max_jumps,
            buffers=_outer_buffers,
            kind='checkpointed',
        )

    def _sample_jump_ops(self, t: Array, psi: Array, key: Array) -> Array:
        # given a state psi at time t that should experience a jump,
        # randomly sample one jump operator from among the provided jump_ops.
        # The probability that a certain jump operator is selected is weighted
        # by the probability that such a jump can occur. For instance for a qubit
        # experiencing amplitude damping, if it is in the ground state then
        # there is probability zero of experiencing an amplitude damping event.

        Ls = stack([L(t) for L in self.Ls])
        Lsd = dag(Ls)
        probs = expect(Lsd @ Ls, psi)
        # for categorical we pass in the log of the probabilities
        logits = jnp.log(jnp.real(probs / (jnp.sum(probs))))
        # randomly sample the index of a single jump operator
        sample_idx = jax.random.categorical(key, logits, shape=(1,))[0]
        return Ls[sample_idx]

    def _average_jump_and_no_jump(
        self, no_jump_state: QArray, jump_state: QArray, no_jump_prob: Array | float
    ) -> QArray:
        no_jump_rho = no_jump_state @ no_jump_state.dag()
        jump_rho = jax.tree.map(
            lambda y: jnp.average(y, axis=0), jump_state @ jump_state.dag()
        )
        return unit(no_jump_prob * no_jump_rho + (1 - no_jump_prob) * jump_rho)


mcsolve_euler_integrator_constructor = partial(
    MCSolveDiffraxIntegrator, diffrax_solver=dx.Euler(), fixed_step=True
)
mcsolve_dopri5_integrator_constructor = partial(
    MCSolveDiffraxIntegrator, diffrax_solver=dx.Dopri5(), fixed_step=False
)
mcsolve_dopri8_integrator_constructor = partial(
    MCSolveDiffraxIntegrator, diffrax_solver=dx.Dopri8(), fixed_step=False
)
mcsolve_tsit5_integrator_constructor = partial(
    MCSolveDiffraxIntegrator, diffrax_solver=dx.Tsit5(), fixed_step=False
)
mcsolve_kvaerno3_integrator_constructor = partial(
    MCSolveDiffraxIntegrator, diffrax_solver=dx.Kvaerno3(), fixed_step=False
)
mcsolve_kvaerno5_integrator_constructor = partial(
    MCSolveDiffraxIntegrator, diffrax_solver=dx.Kvaerno5(), fixed_step=False
)
