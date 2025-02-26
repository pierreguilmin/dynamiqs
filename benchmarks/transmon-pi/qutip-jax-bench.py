import jax
import jax.numpy as jnp
import qutip as qt
import qutip_jax

# global options
qt.settings.core['default_dtype'] = 'jaxdia' # 'jaxdia' or 'jax'

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
a, adag = qt.destroy(3), qt.create(3)
H0 = K * adag @ adag @ a @ a
Hd = a + adag
fds = [
    jax.jit(partial(pulse, T=gate_time, amp=amp, sigma=sigma))
    for amp in amps
    for sigma in sigmas
]
Hs = [[H0, [Hd, fd]] for fd in fds]
c_ops = [jnp.sqrt(kappa) * a]

# initial state
with qt.CoreOptions(default_dtype='jax'):
    psi0 = qt.basis(3, 0)

# options
options = {'method': 'diffrax', 'normalize_output': False}

# function to benchmark
def blocked_mesolve(Hs, *args, **kwargs):
    # `qt.mesolve` does not support `jax.jit`/`jax.vmap` yet: using a for loop instead
    result_list = [qt.mesolve(H, *args, **kwargs).final_state.data._jxa for H in Hs]
    return jnp.asarray(result_list).block_until_ready()

blocked_mesolve(Hs, psi0, tsave, c_ops=c_ops, options=options)
%timeit blocked_mesolve(Hs, psi0, tsave, c_ops=c_ops, options=options)
