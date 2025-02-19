from __future__ import annotations

import equinox as eqx
import jax.tree_util as jtu
from jaxtyping import PyTree, ScalarLike

from ._utils import tree_str_inline
from .progress_meter import AbstractProgressMeter, NoProgressMeter, TqdmProgressMeter
from .qarrays.qarray import QArray

__all__ = ['Options']


class Options(eqx.Module):
    save_states: bool = True
    save_propagators: bool = True
    cartesian_batching: bool = True
    progress_meter: AbstractProgressMeter | None = TqdmProgressMeter()
    t0: ScalarLike | None = None
    save_extra: callable[[QArray], PyTree] | None = None
    max_jumps: int = 100
    nmaxclick: int = 10_000
    smart_sampling: bool = False

    def __init__(
        self,
        save_states: bool = True,
        save_propagators: bool = True,
        cartesian_batching: bool = True,
        progress_meter: AbstractProgressMeter | None = TqdmProgressMeter(),  # noqa: B008
        t0: ScalarLike | None = None,
        save_extra: callable[[QArray], PyTree] | None = None,
        max_jumps: int = 100,
        nmaxclick: int = 10_000,
        smart_sampling: bool = False,
    ):
        if progress_meter is None:
            progress_meter = NoProgressMeter()

        self.save_states = save_states
        self.save_propagators = save_propagators
        self.cartesian_batching = cartesian_batching
        self.progress_meter = progress_meter
        self.t0 = t0
        self.nmaxclick = nmaxclick
        self.smart_sampling = smart_sampling

        # make `save_extra` a valid Pytree with `Partial`
        self.save_extra = jtu.Partial(save_extra) if save_extra is not None else None
        self.max_jumps = max_jumps

    def __str__(self) -> str:
        return tree_str_inline(self)


def check_options(options: Options, solver_name: str):
    supported_options = {
        'sesolve': (
            'save_states',
            'cartesian_batching',
            'progress_meter',
            't0',
            'save_extra',
        ),
        'mesolve': (
            'save_states',
            'cartesian_batching',
            'progress_meter',
            't0',
            'save_extra',
        ),
        'mcsolve': (
            'save_states',
            'cartesian_batching',
            't0',
            'save_extra',
            'max_jumps',
        ),
        'sepropagator': ('save_propagators', 'progress_meter', 't0', 'save_extra'),
        'mepropagator': ('save_propagators', 'cartesian_batching', 't0', 'save_extra'),
        'floquet': ('progress_meter', 't0'),
        'jssesolve': (
            'save_states',
            'cartesian_batching',
            'save_extra',
            'nmaxclick',
            'smart_sampling',
        ),
        'dssesolve': ('save_states', 'cartesian_batching', 'save_extra'),
        'jsmesolve': ('save_states', 'cartesian_batching', 'save_extra', 'nmaxclick'),
        'dsmesolve': ('save_states', 'cartesian_batching', 'save_extra'),
    }
    valid_options = supported_options[solver_name]

    # check that all attributes are set to their default values except for the ones
    # specified in `valid_options`
    for key, value in options.__dict__.items():
        if key not in valid_options and value != getattr(Options(), key):
            valid_options_str = ', '.join(f'`{x}`' for x in valid_options)
            raise ValueError(
                f'Option `{key}` was set to `{value}` but is not used by '
                f'the solver `dq.{solver_name}()` (valid options: '
                f'{valid_options_str}).'
            )
