import qutip as qt
import numpy as np

# global options
qt.settings.core['default_dtype'] = 'Dia' # 'Dia' or 'Dense'

# parameters
K = 2.0
kappa = 0.1
amps = np.linspace(0.0, 0.3, 20)
sigmas = np.linspace(0.5, 10.0, 20)
gate_time = 20.0
num_tsave = 100

# save times
tsave = np.linspace(0.0, gate_time, num_tsave)

# pulse helper functions
def gaussian(t, T, sigma):
    return np.exp(-((t - 0.5 * T)**2 / (2 * sigma**2)))

def pulse(t, T, amp, sigma):
    return amp * (gaussian(t, T, sigma) - gaussian(0, T, sigma))**2

# operators
a, adag = qt.destroy(3), qt.create(3)
H0 = K * adag @ adag @ a @ a
Hd = a + adag
fds = [partial(pulse, T=gate_time, amp=amp, sigma=sigma) for amp in amps for sigma in sigmas]
Hs = [[H0, [Hd, fd]] for fd in fds]
c_ops = [np.sqrt(kappa) * a]

# initial state
with qt.CoreOptions(default_dtype='Dense'):
    psi0 = qt.basis(3, 0)

# options
options = {'method': 'adams', 'normalize_output': False}

# run benchmark
%timeit [qt.mesolve(H, psi0, tsave, c_ops=c_ops, options=options) for H in Hs]
