import dynamiqs as dq
import jax.numpy as jnp

# global settings
dq.set_layout('dia') # 'dense' or 'dia'
dq.set_device('cpu') # 'cpu' or 'gpu'

# parameters
kappa_2 = 1.0
g_cnot = 0.3
nbar = 4.0
num_tsave = 100
N = 32

# save times
alpha = jnp.sqrt(nbar)
gate_time = jnp.pi / (4 * alpha * g_cnot)
tsave = jnp.linspace(0.0, gate_time, num_tsave)

# operators
ac = dq.tensor(dq.destroy(N), dq.eye(N))
nt = dq.tensor(dq.eye(N), dq.number(N))
i = dq.tensor(dq.eye(N), dq.eye(N))
H = g_cnot * (ac + dq.dag(ac)) @ (nt - nbar * i)
jump_ops = [jnp.sqrt(kappa_2) * (ac @ ac - nbar * i)]

# initial state
plus = dq.unit(dq.coherent(N, alpha) + dq.coherent(N, -alpha))
psi0 = dq.tensor(plus, plus)

# options
options = dq.Options(progress_meter=None)
solver = dq.solver.Tsit5(atol=1e-8, rtol=1e-6)

# run benchmark
def blocked_mesolve(*args, **kwargs):
    return dq.mesolve(*args, **kwargs).states.data.block_until_ready()

blocked_mesolve(H, jump_ops, psi0, tsave, options=options, solver=solver)
%timeit -r1 blocked_mesolve(H, jump_ops, psi0, tsave, options=options, solver=solver)
