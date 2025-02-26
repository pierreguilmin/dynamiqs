using QuantumOptics
using BenchmarkTools

# global options
layout = "sparse" # "dense" or "sparse"

# parameters
K = 1.0
epsilons = LinRange(0.0, 0.5, 20)
kappa = 0.1
alpha = 2.0
num_tsave = 100
N = 32

# save times
gate_time = pi / K
tspan = LinRange(0, gate_time, num_tsave)

# operators
basis = FockBasis(N - 1)
a = layout == "sparse" ? destroy(basis) : dense(destroy(basis))
adag = layout == "sparse" ? create(basis) : dense(create(basis))
Hs = [K * adag^2 * a^2 + eps * (adag + a) for eps in epsilons]
J = [sqrt(kappa) * a]

# initial state
psi0 = coherentstate(basis, alpha)

# run benchmark
@btime [timeevolution.master(tspan, psi0, H, J; abstol=1e-8, reltol=1e-6) for H in Hs];
