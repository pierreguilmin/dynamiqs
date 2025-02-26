using QuantumOptics
using BenchmarkTools

# global options
layout == "sparse" # "sparse" or "dense"

# parameters
omega_1 = 4.0
omega_2 = 6.0
J = 0.4
eps = 0.4
num_tsave = 100

# save times
gate_time = 0.5 * pi * abs(omega_2 - omega_1) / (J * eps)
tspan = LinRange(0, gate_time, num_tsave)

# operators
basis = SpinBasis(1//2)
sz = layout == "sparse" ? sigmaz(basis) : dense(sigmaz(basis))
sp = layout == "sparse" ? sigmap(basis) : dense(sigmap(basis))
sm = layout == "sparse" ? sigmam(basis) : dense(sigmam(basis))
i = layout == "sparse" ? identityoperator(basis) : dense(identityoperator(basis))
omega_d = omega_2 - J^2 / (omega_1 - omega_2)
H0 = 0.5 * omega_1 * tensor(sz, i) + 0.5 * omega_2 * tensor(i, sz) + J * (tensor(sp, sm) + tensor(sm, sp))
Hd = eps * (tensor(sp, i) + tensor(sm, i))
H(t, psi) = H0 + cos(omega_d * t) * Hd

# initial state
psi0 = tensor(spindown(basis), spindown(basis))

# run benchmark
@benchmark timeevolution.schroedinger_dynamic(tspan, psi0, H; abstol=1e-8, reltol=1e-6)
