from __future__ import annotations

import equinox as eqx
from jax import Array
from jaxtyping import PyTree

from .gradient import Gradient
from .options import Options
from .solver import Solver

__all__ = ['SEResult', 'MEResult']


def memory_bytes(x: Array) -> int:
    return x.itemsize * x.size


def memory_str(x: Array) -> str:
    mem = memory_bytes(x)
    if mem < 1024**2:
        return f'{mem / 1024:.2f} Kb'
    elif mem < 1024**3:
        return f'{mem / 1024**2:.2f} Mb'
    else:
        return f'{mem / 1024**3:.2f} Gb'


def array_str(x: Array | None) -> str | None:
    return None if x is None else f'Array {x.dtype} {tuple(x.shape)} | {memory_str(x)}'


# the Saved object holds quantities saved during the equation integration
class Saved(eqx.Module):
    ysave: Array
    Esave: Array | None
    extra: PyTree | None


class Result(eqx.Module):
    """Result of the integration.

    Attributes:
        states _(Array)_: Saved states.
        final_state _(Array)_: Saved final state
        expects _(Array, optional)_: Saved expectation values.
        extra _(PyTree, optional)_: Extra data saved.
        infos _(PyTree, optional)_: Solver-dependent information on the resolution.
        tsave _(Array)_: Times for which results were saved.
        solver _(Solver)_: Solver used.
        gradient _(Gradient)_: Gradient used.
        options _(Options)_: Options used.
        final_time _(Array)_: final solution time
    """

    tsave: Array
    solver: Solver
    gradient: Gradient | None
    options: Options
    _saved: Saved
    final_time: Array
    infos: PyTree | None

    @property
    def states(self) -> Array:
        return self._saved.ysave

    @property
    def final_state(self) -> Array:
        if self.options.save_states:
            return self._saved.ysave[..., -1, :, :]
        else:
            return self._saved.ysave

    @property
    def expects(self) -> Array | None:
        return self._saved.Esave

    @property
    def extra(self) -> PyTree | None:
        return self._saved.extra

    def _str_parts(self) -> dict[str, str]:
        return {
            'Solver  ': type(self.solver).__name__,
            'Gradient': (
                type(self.gradient).__name__ if self.gradient is not None else None
            ),
            'States  ': array_str(self.states),
            'Expects ': array_str(self.expects),
            'Extra   ': (
                eqx.tree_pformat(self.extra) if self.extra is not None else None
            ),
            'Infos   ': self.infos if self.infos is not None else None,
        }

    def __str__(self) -> str:
        parts = self._str_parts()

        # remove None values
        parts = {k: v for k, v in parts.items() if v is not None}

        # pad to align colons
        padding = max(len(k) for k in parts) + 1
        parts_str = '\n'.join(f'{k:<{padding}}: {v}' for k, v in parts.items())
        return f'==== {self.__class__.__name__} ====\n' + parts_str

    def to_qutip(self) -> Result:
        raise NotImplementedError

    def to_numpy(self) -> Result:
        raise NotImplementedError


class MCResult(eqx.Module):
    """Result of Monte Carlo integration

    Attributes:
        no_jump_states _(Array)_: Saved no-jump states.
        final_no_jump_state _(Array)_: Saved final no-jump state
        jump_states _(Array)_: Saved states for jump trajectories
        final_jump_states _(Array)_: Saved final states for jump trajectories
        expects _(Array, optional)_: Saved expectation values.
        extra _(PyTree, optional)_: Extra data saved.
        tsave _(Array)_: Times for which results were saved.
        solver _(Solver)_: Solver used.
        gradient _(Gradient)_: Gradient used.
        options _(Options)_: Options used.
        final_time _(Array)_: final solution time
    """

    tsave: Array
    _no_jump_res: Result
    _jump_res: Result
    no_jump_prob: float

    @property
    def no_jump_states(self) -> Array:
        return self._no_jump_res.states

    @property
    def jump_states(self) -> Array:
        return self._jump_res.states

    @property
    def final_no_jump_state(self) -> Array:
        return self._no_jump_res.final_state

    @property
    def final_jump_states(self) -> Array:
        return self._jump_res.final_state

    @property
    def expects(self) -> Array | None:
        if self._no_jump_res.expects is not None:
            jump_expects = self._jump_res.expects
            #TODO which axis will this be in general, if there is batching?
            no_jump_expects = jnp.mean(self._no_jump_res.expects, axis=-3)
            return self.no_jump_prob * jump_expects + (1 - self.no_jump_prob) * no_jump_expects
        else:
            return None

    @property
    def extra(self) -> PyTree | None:
        raise NotImplementedError

    def __str__(self) -> str:
        parts = {
            'No-jump result': str(self._no_jump_res),
            'Jump result': str(self._jump_res),
            'No-jump states  ': array_str(self.no_jump_states),
            'Jump states  ': array_str(self.jump_states),
            'Expects ': array_str(self.expects) if self.expects is not None else None,
        }
        parts = {k: v for k, v in parts.items() if v is not None}
        parts_str = '\n'.join(f'{k}: {v}' for k, v in parts.items())
        return '==== MCResult ====\n' + parts_str

    def to_qutip(self) -> Result:
        raise NotImplementedError

    def to_numpy(self) -> Result:
        raise NotImplementedError


class SEResult(Result):
    """Result of the Schrödinger equation integration.

    Attributes:
        states _(array of shape (nH?, npsi0?, ntsave, n, 1))_: Saved states.
        expects _(array of shape (nH?, npsi0?, nE, ntsave) or None)_: Saved expectation
            values, if specified by `exp_ops`.
        extra _(PyTree or None)_: Extra data saved with `save_extra()` if
            specified in `options`.
        infos _(PyTree or None)_: Solver-dependent information on the resolution.
        tsave _(array of shape (ntsave,))_: Times for which results were saved.
        solver _(Solver)_: Solver used.
        gradient _(Gradient)_: Gradient used.
        options _(Options)_: Options used.
    """


class MEResult(Result):
    """Result of the Lindblad master equation integration.

    Attributes:
        states _(array of shape (nH?, nrho0?, ntsave, n, n))_: Saved states.
        expects _(array of shape (nH?, nrho0?, nE, ntsave) or None)_: Saved expectation
            values, if specified by `exp_ops`.
        extra _(PyTree or None)_: Extra data saved with `save_extra()` if
            specified in `options`.
        infos _(PyTree or None)_: Solver-dependent information on the resolution.
        tsave _(array of shape (ntsave,))_: Times for which results were saved.
        solver _(Solver)_: Solver used.
        gradient _(Gradient)_: Gradient used.
        options _(Options)_: Options used.
    """
