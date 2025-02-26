import qutip_jax as qjax
import qutip as qt
import jax.numpy as jnp

# global options
qt.settings.core['default_dtype'] = 'jaxdia' # 'jaxdia' or 'jax'

# parameters
num_modes = 12
num_tsave = 100
J = 1.0

# save times
gate_time = 1 / J
tsave = jnp.linspace(0.0, gate_time, num_tsave)

# operators
sigmazs = [qt.tensor(*[qt.qeye(2) if j != m else qt.sigmaz() for j in range(num_modes)]) for m in range(num_modes)]
identity = qt.tensor(*[qt.qeye(2) for _ in range(num_modes)])
H = J * sum([sigmazs[m] @ sigmazs[m + 1] for m in range(num_modes - 1)], 0 * identity)

# initial state
with qt.CoreOptions(default_dtype='jax'):
    psi0 = qt.tensor(*[qt.basis(2, 0) for _ in range(num_modes)])

# options
options = {'method': 'diffrax', 'normalize_output': False}

# function to benchmark
def blocked_sesolve(*args, **kwargs):
    return qt.sesolve(*args, **kwargs).final_state.data._jxa.block_until_ready()

blocked_sesolve(H, psi0, tsave, options=options)
%timeit blocked_sesolve(H, psi0, tsave, options=options)
