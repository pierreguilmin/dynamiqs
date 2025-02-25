import jax.numpy as jnp
import jax.random
import pytest

import dynamiqs as dq

from ..order import TEST_LONG


@pytest.mark.run(order=TEST_LONG)
def test_against_mesolve_oscillator(atol=5e-2):
    # parameters
    ntrajs = 80
    dim = 10

    # solver inputs
    a = dq.destroy(dim)
    H = 0.1 * a.dag() @ a + 0.4 * (a + a.dag())
    jump_ops = [a, 0.3 * a.dag()]
    psi0 = dq.basis(dim, 0)
    tsave = jnp.linspace(0.0, 2.0, 11)
    keys = jax.random.split(jax.random.key(31), num=ntrajs)
    exp_ops = [a.dag() @ a]
    options = dq.Options(progress_meter=None)

    # solve with jssesolve and mesolve
    jsseresult = dq.jssesolve(
        H, jump_ops, psi0, tsave, keys, exp_ops=exp_ops, options=options
    )
    meresult = dq.mesolve(H, jump_ops, psi0, tsave, exp_ops=exp_ops, options=options)

    # compare results on average
    mean_jsse_expects = jnp.mean(jsseresult.expects, axis=0)
    mean_jsse_states = jsseresult.states.todm().sum(axis=0) / ntrajs
    assert jnp.allclose(meresult.expects, mean_jsse_expects, atol=atol)
    assert jnp.allclose(meresult.states.to_jax(), mean_jsse_states.to_jax(), atol=atol)


@pytest.mark.run(order=TEST_LONG)
def test_against_mesolve_qubit(atol=5e-2):
    # parameters
    ntrajs = 40
    omega = 2.0 * jnp.pi
    amp = 0.1 * 2.0 * jnp.pi

    # solver inputs
    def H_func(t):
        return -0.5 * omega * dq.sigmaz() + jnp.cos(omega * t) * amp * dq.sigmax()

    H = dq.timecallable(H_func)
    jump_ops = [0.4 * dq.sigmam()]
    psi0 = [dq.ground(), dq.excited()]
    tsave = jnp.linspace(0, 1.0, 41)
    keys = jax.random.split(jax.random.key(42), num=ntrajs)
    exp_ops = [dq.excited().todm(), dq.ground().todm()]
    options = dq.Options(progress_meter=None)

    # solve with jssesolve and mesolve
    jsseresult = dq.jssesolve(
        H, jump_ops, psi0, tsave, keys=keys, exp_ops=exp_ops, options=options
    )
    meresult = dq.mesolve(H, jump_ops, psi0, tsave, exp_ops=exp_ops, options=options)

    # compare results on average
    mean_jsse_expects = jnp.mean(jsseresult.expects, axis=1)
    mean_jsse_states = jsseresult.states.todm().sum(axis=1) / ntrajs
    assert jnp.allclose(meresult.expects, mean_jsse_expects, atol=atol)
    assert jnp.allclose(meresult.states.to_jax(), mean_jsse_states.to_jax(), atol=atol)
