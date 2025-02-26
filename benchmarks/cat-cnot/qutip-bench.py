import qutip as qt
import numpy as np

# global options
qt.settings.core['default_dtype'] = 'Dia' # 'Dia' or 'Dense'

# parameters
kappa_2 = 1.0
g_cnot = 0.3
nbar = 4.0
num_tsave = 100
N = 32

# save times
alpha = np.sqrt(nbar)
gate_time = np.pi / (4 * alpha * g_cnot)
tsave = np.linspace(0.0, gate_time, num_tsave)

# operators
ac = qt.tensor(qt.destroy(N), qt.qeye(N))
nt = qt.tensor(qt.qeye(N), qt.num(N))
i = qt.tensor(qt.qeye(N), qt.qeye(N))
H = g_cnot * (ac + ac.dag()) @ (nt - nbar * i)
c_ops = [np.sqrt(kappa_2) * (ac @ ac - nbar * i)]

# initial state
with qt.CoreOptions(default_dtype='Dense'):
    plus = (qt.coherent(N, alpha) + qt.coherent(N, -alpha)).unit()
    psi0 = qt.tensor(plus, plus)

# options
options = {'method': 'adams', 'normalize_output': False}

# run benchmark
%timeit -r1 qt.mesolve(H, psi0, tsave, c_ops=c_ops, options=options)
