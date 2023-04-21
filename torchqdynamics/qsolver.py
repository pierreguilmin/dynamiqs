from __future__ import annotations

from abc import ABC, abstractmethod

import torch
from torch import Tensor

from .solver_options import SolverOption
from .solver_utils import bexpect


class QSolver(ABC):
    GRADIENT_ALG = ['autograd']

    def __init__(
        self,
        options: SolverOption,
        y0: Tensor,
        exp_ops: Tensor,
        t_save: Tensor,
        gradient_alg: str | None,
        parameters: tuple[torch.nn.Parameter, ...] | None,
    ):
        """

        Args:
            options:
            y0: Initial quantum state, of shape `(..., m, n)`.
            exp_ops:
            t_save: Times for which results are saved.
            gradient_alg:
            parameters (tuple of nn.Parameter): Parameters w.r.t. compute the gradients.
        """
        self.options = options
        self.y0 = y0
        self.exp_ops = exp_ops
        self.t_save = t_save
        self.gradient_alg = gradient_alg
        self.parameters = parameters

        if gradient_alg is not None and gradient_alg not in self.GRADIENT_ALG:
            raise ValueError(
                f'Gradient algorithm {gradient_alg} is not defined or not yet'
                f' supported by this solver ({type(self)}).'
            )

        self.save_counter = 0

        # initialize save tensors
        batch_sizes, (m, n) = y0.shape[:-2], y0.shape[-2:]

        if self.options.save_states:
            # y_save: (..., len(t_save), m, n)
            self.y_save = torch.zeros(
                *batch_sizes, len(self.t_save), m, n, dtype=y0.dtype, device=y0.device
            )

        if len(self.exp_ops) > 0:
            # exp_save: (..., len(exp_ops), len(t_save))
            self.exp_save = torch.zeros(
                *batch_sizes,
                len(self.exp_ops),
                len(self.t_save),
                dtype=y0.dtype,
                device=y0.device,
            )
        else:
            self.exp_save = torch.empty(
                *batch_sizes, len(self.exp_ops), dtype=y0.dtype, device=y0.device
            )

    def next_tsave(self) -> float:
        return self.t_save[self.save_counter]

    def _save_y(self, y: Tensor):
        if self.options.save_states:
            self.y_save[..., self.save_counter, :, :] = y

    def _save_exp_ops(self, y: Tensor):
        if len(self.exp_ops) > 0:
            self.exp_save[..., self.save_counter] = bexpect(self.exp_ops, y)

    def save(self, y: Tensor):
        self._save_y(y)
        self._save_exp_ops(y)
        self.save_counter += 1

    def save_final(self, y: Tensor):
        if not self.options.save_states:
            self.y_save = y

    @abstractmethod
    def run(self):
        pass
