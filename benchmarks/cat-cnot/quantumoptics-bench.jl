using QuantumOptics
using BenchmarkTools

# global options
layout == "sparse" # "sparse" or "dense"

# parameters
kappa_2 = 1.0
g_cnot = 0.3
nbar = 4.0
num_tsave = 100
N = 32

# save times
alpha = sqrt(nbar)
gate_time = pi / (4 * alpha * g_cnot)
tspan = LinRange(0, gate_time, num_tsave)

# operators
basis = FockBasis(N - 1)
a = layout == "sparse" ? destroy(basis) : dense(destroy(basis))
adag = layout == "sparse" ? create(basis) : dense(create(basis))
n = layout == "sparse" ? number(basis) : dense(number(basis))
i = layout == "sparse" ? identityoperator(basis) : dense(identityoperator(basis))
H = g_cnot * tensor(a + adag, n - nbar * i)
J = [sqrt(kappa_2) * tensor(a^2 - nbar * i, i)]

# initial state
plus = normalize(coherentstate(basis, alpha) + coherentstate(basis, -alpha))
psi0 = tensor(plus, plus)

# run benchmark
@btime timeevolution.master(tspan, psi0, H, J; abstol=1e-8, reltol=1e-6);
