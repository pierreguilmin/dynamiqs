from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array
from jaxtyping import PyTree

from .. import PropagatorResult, expm, eye
from ..result import Saved
from ..time_array import ConstantTimeArray, PWCTimeArray, SummedTimeArray
from .abstract_solver import BaseSolver


class ExpmSolver(BaseSolver):

    def __init__(self, *args):
        super().__init__(*args)

        # check that Hamiltonian is either time-independent, piecewise constant
        # or a sum of such Hamiltonians
        if isinstance(self.H, (ConstantTimeArray, PWCTimeArray)):
            constant_or_pwc_check = True
        elif isinstance(self.H, SummedTimeArray):
            constant_or_pwc_check = all(
                isinstance(timearray, (ConstantTimeArray, PWCTimeArray))
                for timearray in self.H.timearrays
            )
        else:
            constant_or_pwc_check = False
        if not constant_or_pwc_check:
            raise TypeError(
                "Solver `Expm` requires a time-independent Hamiltonian, "
                "piece-wise constant Hamiltonian or sum of such Hamiltonians."
            )

    def run(self) -> PyTree:
        # for a constant Hamiltonian, we only need to compute matrix exponentials
        # at the asked-for times
        if isinstance(self.H, ConstantTimeArray):
            times = jnp.concatenate((jnp.asarray(self.t0).reshape(-1), self.ts))
        # for a pwc Hamiltonian, we need to evaluate the matrix exponential
        # for each pwc region, and moreover the times defining those regions may not
        # coincide with the times specified in self.ts. So we need to evaluate the
        # matrix exponential for all such regions
        elif isinstance(self.H, PWCTimeArray):
            times = jnp.sort(
                jnp.concatenate(
                    (jnp.asarray(self.t0).reshape(-1), self.H.times, self.ts)
                )
            )
        # must be summed time array of constant or pwc Hamiltonians
        else:
            times = jnp.sort(
                jnp.concatenate(
                    (jnp.asarray(self.t0).reshape(-1), self.H.discontinuity_ts, self.ts)
                )
            )
        _t_diffs = jnp.diff(times)
        # for times before t0, don't want to include in the propagator calculation
        t_diffs = jnp.where(times[:-1] < self.t0, 0.0, _t_diffs)
        # don't need the last time in times since the hamiltonian is guaranteed
        # to be constant over the region times[-2] to times[-1]
        H_at_ts = jnp.stack([self.H(t) for t in times[:-1]])
        # put the t dimension first, since scan works over the first dimension
        Ht = jnp.expand_dims(t_diffs, jnp.arange(-H_at_ts.ndim + 1, 0)) * H_at_ts
        step_propagators = expm(-1j * Ht)

        def _reduce(prev_prop: Array, next_prop: Array) -> Array:
            # notice the ordering of prev_prop and next_prop, want
            # next_prop to be to the left of prev_prop
            total_prop = next_prop @ prev_prop
            return total_prop, total_prop

        eye_broadcast = jnp.broadcast_to(eye(self.H.shape[-1]), self.H.shape)
        _, propagators_for_times = jax.lax.scan(
            _reduce, eye_broadcast, step_propagators
        )

        # extract the propagators at the correct times
        # the -1 is because the indices of the propagators are defined by t_diffs,
        # not times itself
        t_idxs = jnp.argmin(jnp.abs(times - self.ts[:, None]), axis=1)
        t_idxs = jnp.where(t_idxs > 0, t_idxs - 1, t_idxs)
        propagators = propagators_for_times[t_idxs]
        # note that we can't take the output of scan as final_prop, because
        # it could correspond to a time window of H.times. However
        # the final element of propagators will correspond to the propagator
        # at the final time
        final_prop = propagators[-1]
        propagators = jnp.einsum("t...ij->...tij", propagators)
        saved = Saved(propagators, None, None)
        saved = self.collect_saved(saved, final_prop)
        return self.result(saved)

    def result(self, saved: Saved, infos: PyTree | None = None) -> PropagatorResult:
        return PropagatorResult(
            self.ts, self.solver, self.gradient, self.options, saved, infos
        )
