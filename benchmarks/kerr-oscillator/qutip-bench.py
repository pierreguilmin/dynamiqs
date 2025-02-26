import qutip as qt
import numpy as np

# global options
qt.settings.core['default_dtype'] = 'Dia' # 'Dia' or 'Dense'

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
with qt.CoreOptions(default_dtype='Dense'):
    psi0 = qt.coherent(N, alpha)

# options
options = {'method': 'adams', 'normalize_output': False}

# run benchmark
%timeit [qt.mesolve(H, psi0, tsave, c_ops=c_ops, options=options) for H in Hs]
