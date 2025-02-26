using QuantumOptics
using BenchmarkTools

# global options
layout = "sparse" # "dense" or "sparse"

# parameters
K = 2.0
kappa = 0.1
amps = LinRange(0.0, 0.3, 20)
sigmas = LinRange(0.5, 10.0, 20)
gate_time = 20.0
num_tsave = 100

# save times
tspan = LinRange(0, gate_time, num_tsave)

# pulse helper functions
function gaussian(t, T, sigma)
    return exp(-((t - 0.5 * T)^2 / (2 * sigma^2)))
end

function pulse(t, T, amp, sigma)
    return amp * (gaussian(t, T, sigma) - gaussian(0, T, sigma))^2
end

# operators
basis = FockBasis(2)
a = layout == "sparse" ? destroy(basis) : dense(destroy(basis))
adag = layout == "sparse" ? create(basis) : dense(create(basis))
H0 = K * adag^2 * a^2
Hd = a + adag
J = [sqrt(kappa) * a]
Jdagger = [sqrt(kappa) * adag]
fs = [(t, rho) -> (H0 + pulse(t, gate_time, amp, sigma) * Hd, J, Jdagger) for amp in amps for sigma in sigmas]

# initial state
psi0 = fockstate(basis, 0)

# run benchmark
@benchmark [timeevolution.master_dynamic(tspan, psi0, f; abstol=1e-8, reltol=1e-6) for f in fs]
