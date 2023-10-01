from __future__ import annotations

from abc import ABC, abstractmethod
from time import time

import torch
from torch import Tensor

from .options import Options
from .result import Result
from .utils.td_tensor import TDTensor
from .utils.utils import bexpect


class Solver(ABC):
    def __init__(
        self,
        H: TDTensor,
        y0: Tensor,
        tsave: Tensor,
        exp_ops: Tensor,
        options: Options,
    ):
        """

        Args:
            H:
            y0: Initial quantum state, of shape `(..., m, n)`.
            tsave: Times for which results are saved.
            exp_ops:
            options:
        """
        self.H = H
        self.y0 = y0
        self.tsave = tsave
        self.exp_ops = exp_ops
        self.options = options

        # aliases
        self.cdtype = self.options.cdtype
        self.rdtype = self.options.rdtype
        self.device = self.options.device

        # initialize save tensors
        batch_sizes, (m, n) = y0.shape[:-2], y0.shape[-2:]

        if self.options.save_states:
            # ysave: (..., len(tsave), m, n)
            ysave = torch.zeros(
                *batch_sizes, len(tsave), m, n, dtype=self.cdtype, device=self.device
            )
        else:
            ysave = None

        if len(self.exp_ops) > 0:
            # exp_save: (..., len(exp_ops), len(tsave))
            exp_save = torch.zeros(
                *batch_sizes,
                len(exp_ops),
                len(tsave),
                dtype=self.cdtype,
                device=self.device,
            )
        else:
            exp_save = None

        self.result = Result(options, ysave, tsave, exp_save)

        # initialize save logic and save initial state if necessary
        self._init_save()

    def run(self):
        self.result.start_time = time()
        self._run()
        self.result.end_time = time()

    def _run(self):
        if self.options.gradient_alg is None:
            self.run_nograd()

    @abstractmethod
    def run_nograd(self):
        pass

    def _init_save(self):
        """Initialize the save logic and save the initial state if necessary."""
        self.tsave_counter = 0
        if self.tsave[0] == 0.0:
            self.save(self.y0)

    def save(self, y: Tensor):
        self._save_y(y)
        self._save_exp_ops(y)
        self.tsave_counter += 1

    def _save_y(self, y: Tensor):
        if self.options.save_states:
            self.result.ysave[..., self.tsave_counter, :, :] = y
        # otherwise only save the state if it is the final state
        elif self.tsave_counter == len(self.tsave) - 1:
            self.result.ysave = y

    def _save_exp_ops(self, y: Tensor):
        if len(self.exp_ops) > 0:
            self.result.exp_save[..., self.tsave_counter] = bexpect(self.exp_ops, y)

    def tstop(self):
        """Return tsave excluding `t=0.0`."""
        if self.tsave[0] != 0.0:
            return self.tsave.cpu().numpy()
        else:
            return self.tsave[1:].cpu().numpy()

    def tstop_backward(self):
        """Return tsave excluding the final time and including `t=0.0`."""
        tstop = self.tsave[:-1]
        if tstop[0] != 0.0:
            tstop = torch.cat((torch.zeros(1), tstop))
        return tstop.cpu().numpy()


class AutogradSolver(Solver):
    def _run(self):
        super()._run()
        if self.options.gradient_alg == 'autograd':
            self.run_autograd()

    def run_nograd(self):
        with torch.inference_mode():
            self.run_autograd()

    @abstractmethod
    def run_autograd(self):
        pass


class AdjointSolver(AutogradSolver):
    def _run(self):
        super()._run()
        if self.options.gradient_alg == 'adjoint':
            self.run_adjoint()

    @abstractmethod
    def run_adjoint(self):
        """Integrate an ODE using the adjoint method in the backward pass."""
        pass
