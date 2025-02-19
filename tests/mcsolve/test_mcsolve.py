import jax.numpy as jnp
import jax.random

import dynamiqs as dq
from dynamiqs import Options, mcsolve, mesolve, timecallable


def test_against_mesolve_oscillator(ysave_atol=5e-2):
    num_traj = 40
    options = Options(progress_meter=None)
    dim = 4
    a = dq.destroy(dim)
    y0 = dq.basis(dim, 0)
    jump_ops = [0.1 * a]
    exp_ops = [dq.dag(a) @ a]
    H0 = 0.1 * dq.dag(a) @ a + 0.05 * (a + dq.dag(a))
    tsave = jnp.linspace(0.0, 100.0, 41)

    mcresult = mcsolve(
        H0,
        jump_ops,
        y0,
        tsave,
        keys=jax.random.split(jax.random.key(31), num=num_traj),
        exp_ops=exp_ops,
        options=options,
        root_finder=None,
    )
    meresult = mesolve(H0, jump_ops, y0, tsave, exp_ops=exp_ops, options=options)
    assert jnp.allclose(meresult.expects, mcresult.expects, atol=ysave_atol)
    assert jnp.all(1 - dq.fidelity(meresult.states, mcresult.states) <= ysave_atol)


def test_against_mesolve_qubit(ysave_atol=5e-2):
    num_traj = 40
    options = dq.Options(progress_meter=None)
    initial_states = [dq.basis(2, 1)]
    omega = 2.0 * jnp.pi * 1.0
    amp = 2.0 * jnp.pi * 0.1
    tsave = jnp.linspace(0, 10.0, 41)
    jump_ops = [0.4 * dq.basis(2, 0) @ dq.tobra(dq.basis(2, 1))]
    exp_ops = [
        dq.basis(2, 0) @ dq.tobra(dq.basis(2, 0)),
        dq.basis(2, 1) @ dq.tobra(dq.basis(2, 1)),
    ]

    def H_func(t):
        return -0.5 * omega * dq.sigmaz() + jnp.cos(omega * t) * amp * dq.sigmax()

    mcresult = dq.mcsolve(
        timecallable(H_func),
        jump_ops,
        initial_states,
        tsave,
        keys=jax.random.split(jax.random.key(4242434), num=num_traj),
        exp_ops=exp_ops,
        options=options,
        root_finder=None,
    )
    meresult = dq.mesolve(
        timecallable(H_func), jump_ops, initial_states, tsave, exp_ops=exp_ops
    )
    assert jnp.allclose(meresult.expects, mcresult.expects, atol=ysave_atol)
    assert jnp.all(1 - dq.fidelity(meresult.states, mcresult.states) <= ysave_atol)
