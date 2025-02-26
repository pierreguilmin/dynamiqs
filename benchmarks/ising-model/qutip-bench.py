import qutip as qt
import numpy as np

# global options
qt.settings.core['default_dtype'] = 'Dia' # 'Dia' or 'Dense'

# parameters
num_modes = 12
num_tsave = 100
J = 1.0

# save times
gate_time = 1 / J
tsave = np.linspace(0.0, gate_time, num_tsave)

# operators
sigmazs = [qt.tensor(*[qt.qeye(2) if j != m else qt.sigmaz() for j in range(num_modes)]) for m in range(num_modes)]
identity = qt.tensor(*[qt.qeye(2) for _ in range(num_modes)])
H = J * sum([sigmazs[m] @ sigmazs[m + 1] for m in range(num_modes - 1)], 0 * identity)

# initial state
with qt.CoreOptions(default_dtype='Dense'):
    psi0 = qt.tensor(*[qt.basis(2, 0) for _ in range(num_modes)])

# options
options = {'method': 'adams', 'normalize_output': False}

# run benchmark
%timeit qt.sesolve(H, psi0, tsave, options=options)
