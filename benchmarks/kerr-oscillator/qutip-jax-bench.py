import qutip_jax
import qutip as qt
import jax.numpy as jnp

# global options
qt.settings.core['default_dtype'] = 'jaxdia' # 'jaxdia' or 'jax'

# parameters
K = 1.0
epsilons = np.linspace(0.0, 0.5, 20)
kappa = 0.1
alpha = 2.0
num_tsave = 100
N = 32

# save times
gate_time = np.pi / K
tsave = np.linspace(0.0, gate_time, num_tsave)

# operators
a, adag = qt.destroy(N), qt.create(N)
Hs = [K * adag @ adag @ a @ a + eps * (a + adag) for eps in epsilons]
c_ops = [np.sqrt(kappa) * a]

# initial state
with qt.CoreOptions(default_dtype='jax'):
    psi0 = qt.coherent(N, alpha)

# options
options = {'method': 'diffrax', 'normalize_output': False}

# function to benchmark
def blocked_mesolve(Hs, *args, **kwargs):
    # `qt.mesolve` does not support `jax.jit`/`jax.vmap` yet: using a for loop instead
    result_list = [qt.mesolve(H, *args, **kwargs).final_state.data._jxa for H in Hs]
    return jnp.asarray(result_list).block_until_ready()

blocked_mesolve(Hs, psi0, tsave, c_ops=c_ops, options=options)
%timeit blocked_mesolve(Hs, psi0, tsave, c_ops=c_ops, options=options)
