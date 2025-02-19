from __future__ import annotations

from abc import abstractmethod

import equinox as eqx
from jax import Array
from jaxtyping import PRNGKeyArray, PyTree

from .gradient import Gradient
from .method import Method
from .options import Options
from .qarrays.qarray import QArray
from .qarrays.utils import to_jax

__all__ = [
    'FloquetResult',
    'MCSolveResult',
    'MEPropagatorResult',
    'JSSESolveResult',
    'DSSESolveResult',
    'JSMESolveResult',
    'MESolveResult',
    'DSMESolveResult',
    'SEPropagatorResult',
    'SESolveResult',
]


def memory_bytes(x: Array) -> int:
    return x.itemsize * x.size


def memory_str(x: Array) -> str:
    mem = memory_bytes(x)
    if mem < 1024**2:
        return f'{mem / 1024:.1f} Kb'
    elif mem < 1024**3:
        return f'{mem / 1024**2:.1f} Mb'
    else:
        return f'{mem / 1024**3:.1f} Gb'


def array_str(x: Array | QArray | None) -> str | None:
    # TODO: implement memory_str for `QArray` rather than converting to JAX array
    if x is None:
        return None
    type_name = 'Array' if isinstance(x, Array) else 'QArray'
    x = to_jax(x)
    return f'{type_name} {x.dtype} {tuple(x.shape)} | {memory_str(x)}'


# the Saved object holds quantities saved during the equation integration
class Saved(eqx.Module):
    ysave: QArray
    extra: PyTree | None


class SolveSaved(Saved):
    Esave: Array | None


class DiffusiveSolveSaved(SolveSaved):
    Isave: Array


class PropagatorSaved(Saved):
    pass


class FloquetSaved(Saved):
    quasienergies: Array


class Result(eqx.Module):
    tsave: Array
    method: Method
    gradient: Gradient | None
    options: Options
    _saved: Saved
    infos: PyTree | None

    @property
    def extra(self) -> PyTree | None:
        return self._saved.extra

    def to_qutip(self) -> Result:
        raise NotImplementedError

    def to_numpy(self) -> Result:
        raise NotImplementedError

    def block_until_ready(self) -> Result:
        _ = self._saved.ysave.block_until_ready()
        return self

    def _str_parts(self) -> dict[str, str | None]:
        return {
            'Method': type(self.method).__name__,
            'Gradient': (
                type(self.gradient).__name__ if self.gradient is not None else None
            ),
            'Infos': self.infos if self.infos is not None else None,
            'Extra': (eqx.tree_pformat(self.extra) if self.extra is not None else None),
        }

    def __str__(self) -> str:
        parts = self._str_parts()

        # remove None values
        parts = {k: v for k, v in parts.items() if v is not None}

        # pad to align colons
        padding = max(len(k) for k in parts) + 1
        parts_str = '\n'.join(f'{k:<{padding}}: {v}' for k, v in parts.items())
        return f'==== {self.__class__.__name__} ====\n' + parts_str

    @classmethod
    def out_axes(cls) -> SolveResult:
        return cls(None, None, None, None, 0, 0)


class SolveResult(Result):
    @property
    def states(self) -> QArray:
        return self._saved.ysave

    @property
    def final_state(self) -> QArray:
        return self.states[..., -1, :, :]

    @property
    def expects(self) -> Array | None:
        return self._saved.Esave

    def _str_parts(self) -> dict[str, str | None]:
        d = super()._str_parts()
        return d | {
            'States': array_str(self.states),
            'Expects': array_str(self.expects),
        }


class PropagatorResult(Result):
    @property
    def propagators(self) -> QArray:
        return self._saved.ysave

    @property
    def final_propagator(self) -> QArray:
        return self.propagators[..., -1, :, :]

    def _str_parts(self) -> dict[str, str | None]:
        d = super()._str_parts()
        return d | {'Propagators': array_str(self.propagators)}


class FloquetResult(Result):
    T: float

    @property
    def modes(self) -> QArray:
        return self._saved.ysave

    @property
    def quasienergies(self) -> Array:
        return self._saved.quasienergies

    def _str_parts(self) -> dict[str, str | None]:
        d = super()._str_parts()
        return d | {
            'Modes': array_str(self.modes),
            'Quasienergies': array_str(self.quasienergies),
        }

    @classmethod
    def out_axes(cls) -> SolveResult:
        return cls(None, None, None, None, 0, 0, None)


class SESolveResult(SolveResult):
    pass


class MESolveResult(SolveResult):
    pass


class MCJumpResult(SolveResult):
    jump_times: Array
    num_jumps: Array


class MCNoJumpResult(SolveResult):
    no_jump_prob: Array


class MCSolveResult(SolveResult):
    _no_jump_result: MCNoJumpResult
    _jump_result: MCJumpResult

    @property
    def no_jump_states(self) -> QArray:
        return self._no_jump_result.states

    @property
    def jump_states(self) -> QArray:
        return self._jump_result.states

    @property
    def final_no_jump_state(self) -> QArray:
        return self._no_jump_result.final_state

    @property
    def final_jump_states(self) -> QArray:
        return self._jump_result.final_state

    @property
    def no_jump_prob(self) -> Array:
        return self._no_jump_result.no_jump_prob

    @property
    def jump_times(self) -> Array:
        return self._jump_result.jump_times

    @property
    def num_jumps(self) -> Array:
        return self._jump_result.jump_times

    def __str__(self) -> str:
        parts = {
            'No-jump states': array_str(self.no_jump_states),
            'Jump states': array_str(self.jump_states),
            'No-jump probability  ': array_str(self.no_jump_prob),
            'Jump times  ': array_str(self.jump_times),
            'Number of jumps in each trajectory  ': array_str(self.num_jumps),
            'Infos': self.infos if self.infos is not None else None,
            'Expects ': array_str(self.expects) if self.expects is not None else None,
        }
        parts = {k: v for k, v in parts.items() if v is not None}
        parts_str = '\n'.join(f'{k}: {v}' for k, v in parts.items())
        return '==== MCResult ====\n' + parts_str

    @classmethod
    def out_axes(cls) -> SolveResult:
        return cls(None, None, None, None, 0, 0, 0, 0)


class SEPropagatorResult(PropagatorResult):
    pass


class MEPropagatorResult(PropagatorResult):
    pass


class JSSESolveResult(SolveResult):
    @abstractmethod
    def no_jump_state(self) -> Array | None:
        pass

    @abstractmethod
    def no_jump_proba(self) -> Array | None:
        pass


class JSMESolveResult(SolveResult):
    pass


class DiffusiveSolveResult(SolveResult):
    keys: PRNGKeyArray

    @property
    def measurements(self) -> Array:
        return self._saved.Isave

    def _str_parts(self) -> dict[str, str | None]:
        d = super()._str_parts()
        return d | {'Measurements': array_str(self.measurements)}

    @classmethod
    def out_axes(cls) -> SolveResult:
        return cls(None, None, None, None, 0, 0, 0)


class DSSESolveResult(DiffusiveSolveResult):
    pass


class DSMESolveResult(DiffusiveSolveResult):
    pass
