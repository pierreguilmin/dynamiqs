import qutip_jax as qjax
import qutip as qt
import jax.numpy as jnp

# global options
qt.settings.core['default_dtype'] = 'jaxdia' # 'jaxdia' or 'jax'

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
ac = qt.tensor(qt.destroy(N), qt.qeye(N))
nt = qt.tensor(qt.qeye(N), qt.num(N))
i = qt.tensor(qt.qeye(N), qt.qeye(N))
H = g_cnot * (ac + ac.dag()) @ (nt - nbar * i)
c_ops = [jnp.sqrt(kappa_2) * (ac @ ac - nbar * i)]

# initial state
with qt.CoreOptions(default_dtype='jax'):
    plus = (qt.coherent(N, alpha) + qt.coherent(N, -alpha)).unit()
    psi0 = qt.tensor(plus, plus)

# options
options = {'method': 'diffrax', 'normalize_output': False}

# function to benchmark
def blocked_mesolve(*args, **kwargs):
    return qt.mesolve(*args, **kwargs).final_state.data._jxa.block_until_ready()

blocked_mesolve(H, psi0, tsave, c_ops=c_ops, options=options)
%timeit -r1 blocked_mesolve(H, psi0, tsave, c_ops=c_ops, options=options)
