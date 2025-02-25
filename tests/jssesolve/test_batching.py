import jax
import jax.numpy as jnp
import pytest

import dynamiqs as dq

from ..order import TEST_LONG


def rand_jssesolve_args(n, nH, nLs, npsi0, nEs):
    nkeys = len(nLs) + 4
    kH, *kLs, kpsi0, kEs, kmc = jax.random.split(jax.random.key(31), nkeys)
    H = dq.random.herm(kH, (*nH, n, n))
    Ls = [dq.random.herm(kL, (*nL, n, n)) for kL, nL in zip(kLs, nLs, strict=False)]
    psi0 = dq.random.ket(kpsi0, (*npsi0, n, 1))
    Es = dq.random.complex(kEs, (nEs, n, n))
    return H, Ls, psi0, Es, kmc


@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('nH', [(), (3,), (3, 4)])
@pytest.mark.parametrize('npsi0', [(), (5,)])
@pytest.mark.parametrize('nL', [(), (6,)])
def test_cartesian_batching(nH, npsi0, nL):
    n = 2
    nLs = [nL]
    ntrajs = 7
    nEs = 8
    ntsave = 9

    # run jssesolve
    H, Ls, psi0, Es, kmc = rand_jssesolve_args(n, nH, nLs, npsi0, nEs)
    keys = jax.random.split(kmc, num=ntrajs)
    tsave = jnp.linspace(0.0, 0.1, ntsave)
    options = dq.Options(progress_meter=None)
    result = dq.jssesolve(H, Ls, psi0, tsave, keys=keys, exp_ops=Es, options=options)

    # check result shape
    assert result.states.shape == (*nH, *nL, *npsi0, ntrajs, ntsave, n, 1)
    assert result.expects.shape == (*nH, *nL, *npsi0, ntrajs, nEs, ntsave)


# H has fixed shape (3, 4, n, n) for the next test case, we test a broad ensemble of
# compatible broadcastable shape
@pytest.mark.run(order=TEST_LONG)
@pytest.mark.parametrize('nL1', [(), (4, 1, 3)])
@pytest.mark.parametrize('npsi0', [(), (1,), (3,), (2, 1), (2, 3), (4, 1, 3)])
@pytest.mark.parametrize('ntrajs', [(1,), (5,), (5, 6)])
def test_flat_batching(nL1, npsi0, ntrajs):
    n = 2
    nH = (2, 3)
    nLs = [nL1, ()]
    nEs = 7
    ntsave = 8

    # run jssesolve
    H, Ls, psi0, Es, kmc = rand_jssesolve_args(n, nH, nLs, npsi0, nEs)
    keys = jax.random.split(kmc, num=ntrajs)
    tsave = jnp.linspace(0.0, 0.1, ntsave)
    options = dq.Options(progress_meter=None, cartesian_batching=False)
    result = dq.jssesolve(H, Ls, psi0, tsave, keys=keys, exp_ops=Es, options=options)

    # check result shape
    broadcast_shape = jnp.broadcast_shapes(nH, nL1, npsi0)
    assert result.states.shape == (*broadcast_shape, *ntrajs, ntsave, n, 1)
    assert result.expects.shape == (*broadcast_shape, *ntrajs, nEs, ntsave)
