from functools import partial

import dynamiqs as dq
import jax.numpy as jnp

# global settings
dq.set_layout('dia') # 'dense' or 'dia'
dq.set_device('cpu') # 'cpu' or 'gpu'

# parameters
K = 2.0
kappa = 0.1
amps = jnp.linspace(0.0, 0.3, 20)
sigmas = jnp.linspace(0.5, 10.0, 20)
gate_time = 20.0
num_tsave = 100

# save times
tsave = jnp.linspace(0.0, gate_time, num_tsave)

# pulse helper functions
def gaussian(t, T, sigma):
    return jnp.exp(-((t - 0.5 * T)**2 / (2 * sigma**2)))

def pulse(t, T, amp, sigma):
    return amp * (gaussian(t, T, sigma) - gaussian(0, T, sigma))**2

# operators
a, adag = dq.destroy(3), dq.create(3)
H0 = K * adag @ adag @ a @ a
Hd = a + adag
fd = partial(pulse, T=gate_time, amp=amps[:, None], sigma=sigmas[None, :])
Hs = H0 + dq.modulated(fd, Hd)
jump_ops = [jnp.sqrt(kappa) * a]

# initial state
psi0 = dq.basis(3, 0)

# options
options = dq.Options(progress_meter=None)
solver = dq.solver.Tsit5(atol=1e-8, rtol=1e-6)

# run benchmark
def blocked_mesolve(*args, **kwargs):
    return dq.mesolve(*args, **kwargs).states.data.block_until_ready()

blocked_mesolve(Hs, jump_ops, psi0, tsave, options=options, solver=solver)
%timeit blocked_mesolve(Hs, jump_ops, psi0, tsave, options=options, solver=solver)
