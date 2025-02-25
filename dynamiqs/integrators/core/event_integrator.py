from __future__ import annotations

from functools import partial

import diffrax as dx
import equinox as eqx
import jax
import jax.numpy as jnp
from equinox.internal import while_loop
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree, Scalar

from ...qarrays.qarray import QArray
from ...qarrays.utils import stack
from ...result import Result
from ...utils.general import dag, expect, norm, unit
from .abstract_integrator import StochasticBaseIntegrator
from .diffrax_integrator import DiffraxIntegrator
from .interfaces import JSSEInterface, SolveInterface
from .save_mixin import JumpSolveSaveMixin


class JSSEInnerState(eqx.Module):
    saved: PyTree
    save_index: int


class JSSEState(eqx.Module):
    y: QArray  # quantum state
    t: Scalar  # time
    key: PRNGKeyArray  # active key
    numclicks: int  # number of clicks
    clicktimes: Array  # time of clicks
    inner_state: JSSEInnerState  # saved quantities


def save_buffers(inner_state: JSSEInnerState) -> PyTree:
    assert type(inner_state) is JSSEInnerState
    return inner_state.saved


def loop_buffers(state: JSSEState) -> PyTree:
    assert type(state) is JSSEState
    return state.inner_state.saved


class JSSESolveEventIntegrator(
    StochasticBaseIntegrator,
    DiffraxIntegrator,
    JSSEInterface,
    SolveInterface,
    JumpSolveSaveMixin,
):
    """Integrator computing the time evolution of the Jump SSE using Diffrax events."""

    @property
    def terms(self) -> dx.AbstractTerm:
        def vector_field(t, y, _):  # noqa: ANN001, ANN202
            L, H = self.L(t), self.H(t)
            return -1j * H @ y + sum([-0.5 * _L.dag() @ (_L @ y) for _L in L])

        return dx.ODETerm(vector_field)

    def run(self) -> Result:
        def loop_condition(state: JSSEState) -> bool:
            return state.t < self.t1

        def loop_body(state: JSSEState) -> JSSEState:
            # pick a random number for the next detection event
            key, click_key, jump_key = jax.random.split(state.key, num=3)
            rand = jax.random.uniform(click_key)

            # solve until the next detection event
            solution = self._solve_until_click(state.y, state.t, rand)
            temp_saved = solution.ys[0]
            yclick = solution.ys[1][0]
            tclick = solution.ts[1][0]

            # if the event triggered, update the quantum state and clicktimes
            # else, the final time was reached, and do nothing
            def event(
                yclick: QArray, clicktimes: Array, numclicks: int
            ) -> tuple[QArray, Array, int]:
                # find a random jump operator among the provided jump_ops, and apply it
                jump_op, idx = self._sample_jump_ops(tclick, yclick, jump_key)
                yclick = unit(jump_op @ yclick)

                # update clicktimes
                clicktimes = clicktimes.at[idx, state.numclicks].set(tclick)
                numclicks += 1

                return yclick, clicktimes, numclicks

            def skip(
                yclick: QArray, clicktimes: Array, numclicks: int
            ) -> tuple[QArray, Array, int]:
                return yclick, clicktimes, numclicks

            yclick, clicktimes, numclicks = jax.lax.cond(
                solution.event_mask,
                event,
                skip,
                yclick,
                state.clicktimes,
                state.numclicks,
            )

            # save intermediate states and expectation values, based on the code
            # in diffrax/_integrate.py which can be found at:
            # https://github.com/patrick-kidger/diffrax/blob/main/diffrax/_integrate.py#L427-L458
            def save_cond(inner_state: JSSEInnerState) -> bool:
                save_index = inner_state.save_index
                return (self.ts[save_index] <= tclick) & (save_index < len(self.ts))

            def save_body(inner_state: JSSEInnerState) -> JSSEInnerState:
                idx = inner_state.save_index
                saved = jax.tree.map(
                    lambda _temp_saved, _saved: _saved.at[idx].set(_temp_saved[idx]),
                    temp_saved,
                    inner_state.saved,
                )
                return JSSEInnerState(saved, idx + 1)

            inner_state = while_loop(
                save_cond,
                save_body,
                state.inner_state,
                max_steps=len(self.ts),
                buffers=save_buffers,
                kind='checkpointed',
            )

            # return updated state
            return JSSEState(yclick, tclick, key, numclicks, clicktimes, inner_state)

        # prepare the initial state to loop over
        y = stack([self.y0] * len(self.ts))
        saved = self.reorder_Esave(self.save(y))
        clicktimes = jnp.full((len(self.Ls), self.options.nmaxclick), jnp.nan)
        inner_state = JSSEInnerState(saved, 0)
        state = JSSEState(self.y0, self.t0, self.key, 0, clicktimes, inner_state)

        # loop over no jump evolutions until the final time is reached
        final_state = while_loop(
            loop_condition,
            loop_body,
            state,
            kind='checkpointed',
            buffers=loop_buffers,
            max_steps=self.options.nmaxclick,
        )

        # collect and return results
        saved = final_state.inner_state.saved  # of type SolveSaved
        clicktimes = final_state.clicktimes
        saved = self.postprocess_saved(saved, clicktimes)  # of type JumpSolveSaved
        return self.result(saved, infos=None)

    def _solve_until_click(self, y0: QArray, t0: Array, rand: Array) -> dx.Solution:
        # === prepare saveat
        fn = lambda t, y, args: self.save(y)  # noqa: ARG005
        ts = jnp.where(self.ts >= t0, self.ts, t0)  # clip ts to start at t0
        subsaveat_a = dx.SubSaveAt(ts=ts, fn=fn)  # save solution regularly
        subsaveat_b = dx.SubSaveAt(t1=True)  # save last state
        saveat = dx.SaveAt(subs=[subsaveat_a, subsaveat_b])

        # === prepare event
        def cond_fn(t: Scalar, y: QArray, *args, **kwargs) -> Array:
            return norm(y) ** 2 - rand

        event = dx.Event(cond_fn, self.method.root_finder)

        # === solve differential equation with diffrax
        return self.diffeqsolve(t0=t0, t1=self.t1, y0=y0, saveat=saveat, event=event)

    def _sample_jump_ops(self, t: Array, psi: Array, key: Array) -> tuple[Array, int]:
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
        return Ls[sample_idx], sample_idx


jssesolve_event_euler_integrator_constructor = partial(
    JSSESolveEventIntegrator, diffrax_solver=dx.Euler(), fixed_step=True
)
jssesolve_event_dopri5_integrator_constructor = partial(
    JSSESolveEventIntegrator, diffrax_solver=dx.Dopri5(), fixed_step=False
)
jssesolve_event_dopri8_integrator_constructor = partial(
    JSSESolveEventIntegrator, diffrax_solver=dx.Dopri8(), fixed_step=False
)
jssesolve_event_tsit5_integrator_constructor = partial(
    JSSESolveEventIntegrator, diffrax_solver=dx.Tsit5(), fixed_step=False
)
jssesolve_event_kvaerno3_integrator_constructor = partial(
    JSSESolveEventIntegrator, diffrax_solver=dx.Kvaerno3(), fixed_step=False
)
jssesolve_event_kvaerno5_integrator_constructor = partial(
    JSSESolveEventIntegrator, diffrax_solver=dx.Kvaerno5(), fixed_step=False
)
