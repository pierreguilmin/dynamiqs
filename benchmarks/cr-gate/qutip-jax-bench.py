import qutip_jax
import qutip as qt
import jax.numpy as jnp
import jax

# global options
qt.settings.core['default_dtype'] = 'jaxdia' # 'jaxdia' or 'jax'

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
sz1 = qt.tensor(qt.sigmaz(), qt.qeye(2))
sz2 = qt.tensor(qt.qeye(2), qt.sigmaz())
sp1 = qt.tensor(qt.sigmap(), qt.qeye(2))
sp2 = qt.tensor(qt.qeye(2), qt.sigmap())
sm1 = qt.tensor(qt.sigmam(), qt.qeye(2))
sm2 = qt.tensor(qt.qeye(2), qt.sigmam())
omega_d = omega_2 - J**2 / (omega_1 - omega_2)
H0 = 0.5 * omega_1 * sz1 + 0.5 * omega_2 * sz2 + J * (sp1 * sm2 + sm1 * sp2)
Hd = eps * (sp1 + sm1)
fd = jax.jit(lambda t: jnp.cos(omega_d * t))
H = [H0, [Hd, fd]]

# initial state
with qt.CoreOptions(default_dtype='jax'):
    psi0 = qt.tensor(qt.basis(2, 1), qt.basis(2, 1))

# options
options = {'method': 'diffrax', 'normalize_output': False}

# function to benchmark
def blocked_sesolve(*args, **kwargs):
    return qt.sesolve(*args, **kwargs).final_state.data._jxa.block_until_ready()

blocked_sesolve(H, psi0, tsave, options=options)
%timeit blocked_sesolve(H, psi0, tsave, options=options)
