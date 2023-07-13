import logging
from abc import ABC

import torch

from dynamiqs.options import Options

from .system import System


class SolverTester(ABC):
    def _test_batching(self, options: Options, system: System):
        """Test the batching of `H` and `y0`, and the returned object sizes."""
        print(system._state_shape)
        m, n = system._state_shape
        n_exp_ops = len(system.exp_ops)
        b_H = len(system.H_batched)
        b_y0 = len(system.y0_batched)
        num_t_save = 11
        t_save = system.t_save(num_t_save)

        run = lambda H, y0: system._run(H, y0, t_save, options)

        # no batching
        result = run(system.H, system.y0)
        assert result.y_save.shape == (num_t_save, m, n)
        assert result.exp_save.shape == (n_exp_ops, num_t_save)

        # batched H
        result = run(system.H_batched, system.y0)
        assert result.y_save.shape == (b_H, num_t_save, m, n)
        assert result.exp_save.shape == (b_H, n_exp_ops, num_t_save)

        # batched y0
        result = run(system.H, system.y0_batched)
        assert result.y_save.shape == (b_y0, num_t_save, m, n)
        assert result.exp_save.shape == (b_y0, n_exp_ops, num_t_save)

        # batched H and y0
        result = run(system.H_batched, system.y0_batched)
        assert result.y_save.shape == (b_H, b_y0, num_t_save, m, n)
        assert result.exp_save.shape == (b_H, b_y0, n_exp_ops, num_t_save)

    def test_batching(self):
        pass

    def _test_correctness(
        self,
        options: Options,
        system: System,
        *,
        num_t_save: int,
        y_save_norm_atol: float = 1e-2,
        exp_save_rtol: float = 1e-2,
        exp_save_atol: float = 1e-2,
    ):
        t_save = system.t_save(num_t_save)
        result = system.run(t_save, options)

        # === test y_save
        errs = torch.linalg.norm(result.y_save - system.states(t_save), dim=(-2, -1))
        assert torch.all(errs <= y_save_norm_atol)

        # === test exp_save
        assert torch.allclose(
            result.exp_save,
            system.expects(t_save),
            rtol=exp_save_rtol,
            atol=exp_save_atol,
        )

    def test_correctness(self):
        pass

    def _test_gradient(
        self,
        options: Options,
        system: System,
        *,
        num_t_save: int,
        rtol: float = 1e-3,
        atol: float = 1e-5,
    ):
        t_save = system.t_save(num_t_save)
        result = system.run(t_save, options)

        # === test gradients depending on final y_save
        loss_state = system.loss_state(result.y_save[-1])
        grads_loss_state = torch.autograd.grad(
            loss_state, system.parameters, retain_graph=True
        )
        grads_loss_state = torch.stack(grads_loss_state)
        true_grads_loss_state = system.grads_loss_state(t_save[-1])

        logging.warning(f'grads_loss_state           = {grads_loss_state}')
        logging.warning(f'true_grads_loss_state      = {true_grads_loss_state}')

        assert torch.allclose(
            grads_loss_state, true_grads_loss_state, rtol=rtol, atol=atol
        )

        # === test gradient depending on final exp_save
        losses_expect = system.losses_expect(result.exp_save[:, -1])
        grads_losses_expect = [
            torch.stack(torch.autograd.grad(loss, system.parameters, retain_graph=True))
            for loss in losses_expect
        ]
        grads_losses_expect = torch.stack(grads_losses_expect)
        true_grads_losses_expect = system.grads_losses_expect(t_save[-1])

        logging.warning(f'grads_losses_expect      = {grads_losses_expect}')
        logging.warning(f'true_grads_losses_expect = {true_grads_losses_expect}')

        assert torch.allclose(
            grads_losses_expect, true_grads_losses_expect, rtol=rtol, atol=atol
        )
