import dynamiqs as dq
import jax.numpy as jnp

# global settings
dq.set_layout('dia') # 'dense' or 'dia'
dq.set_device('cpu') # 'cpu' or 'gpu'

# parameters
num_modes = 12
num_tsave = 100
J = 1.0

# save times
gate_time = 1 / J
tsave = jnp.linspace(0.0, gate_time, num_tsave)

# operators
sigmazs = [
    dq.tensor(*[dq.eye(2) if j != m else dq.sigmaz() for j in range(num_modes)])
    for m in range(num_modes)
]
identity = dq.tensor(*[dq.eye(2) for _ in range(num_modes)])
H = J * sum([sigmazs[m] @ sigmazs[m + 1] for m in range(num_modes - 1)], 0 * identity)

# initial state
psi0 = dq.tensor(*[dq.basis(2, 0) for _ in range(num_modes)])

# options
options = dq.Options(progress_meter=None)
solver = dq.solver.Tsit5(atol=1e-8, rtol=1e-6)

# run benchmark
def blocked_sesolve(*args, **kwargs):
    return dq.sesolve(*args, **kwargs).states.data.block_until_ready()

blocked_sesolve(H, psi0, tsave, options=options, solver=solver)
%timeit blocked_sesolve(H, psi0, tsave, options=options, solver=solver)
