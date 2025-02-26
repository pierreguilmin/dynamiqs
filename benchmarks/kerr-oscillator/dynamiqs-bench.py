import dynamiqs as dq
import jax.numpy as jnp

# global settings
dq.set_layout('dia') # 'dense' or 'dia'
dq.set_device('cpu') # 'cpu' or 'gpu'

# parameters
K = 1.0
epsilons = jnp.linspace(0.0, 0.5, 20)
kappa = 0.1
alpha = 2.0
num_tsave = 100
N = 32

# save times
gate_time = jnp.pi / K
tsave = jnp.linspace(0.0, gate_time, num_tsave)

# operators
a, adag = dq.destroy(N), dq.create(N)
Hs = K * adag @ adag @ a @ a + epsilons[:, None, None] * (a + adag)
jump_ops = [jnp.sqrt(kappa) * a]

# initial state
psi0 = dq.coherent(N, alpha)

# options
options = dq.Options(progress_meter=None)
solver = dq.solver.Tsit5(atol=1e-8, rtol=1e-6)

# run benchmark
def blocked_mesolve(*args, **kwargs):
    return dq.mesolve(*args, **kwargs).states.data.block_until_ready()

blocked_mesolve(Hs, jump_ops, psi0, tsave, options=options, solver=solver)
%timeit blocked_mesolve(Hs, jump_ops, psi0, tsave, options=options, solver=solver)
