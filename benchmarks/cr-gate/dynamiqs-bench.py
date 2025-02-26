import dynamiqs as dq
import jax.numpy as jnp

# global settings
dq.set_layout('dia') # 'dense' or 'dia'
dq.set_device('cpu') # 'cpu' or 'gpu'

# parameters
omega_1 = 4.0
omega_2 = 6.0
J = 0.4
eps = 0.4
num_tsave = 100

# save times
gate_time = 0.5 * jnp.pi * abs(omega_2 - omega_1) / (J * eps)
tsave = jnp.linspace(0.0, gate_time, num_tsave)

# operators
sz1 = dq.tensor(dq.sigmaz(), dq.eye(2))
sz2 = dq.tensor(dq.eye(2), dq.sigmaz())
sp1 = dq.tensor(dq.sigmap(), dq.eye(2))
sp2 = dq.tensor(dq.eye(2), dq.sigmap())
sm1 = dq.tensor(dq.sigmam(), dq.eye(2))
sm2 = dq.tensor(dq.eye(2), dq.sigmam())
omega_d = omega_2 - J**2 / (omega_1 - omega_2)
H0 = 0.5 * omega_1 * sz1 + 0.5 * omega_2 * sz2 + J * (sp1 @ sm2 + sm1 @ sp2)
Hd = eps * (sp1 + sm1)
fd = lambda t: jnp.cos(omega_d * t)  # noqa: E731
H = H0 + dq.modulated(fd, Hd)

# initial state
psi0 = dq.tensor(dq.basis(2, 1), dq.basis(2, 1))

# options
options = dq.Options(progress_meter=None)
solver = dq.solver.Tsit5(atol=1e-8, rtol=1e-6)

# run benchmark
def blocked_sesolve(*args, **kwargs):
    return dq.sesolve(*args, **kwargs).states.data.block_until_ready()

blocked_sesolve(H, psi0, tsave, options=options, solver=solver)
%timeit blocked_sesolve(H, psi0, tsave, options=options, solver=solver)
