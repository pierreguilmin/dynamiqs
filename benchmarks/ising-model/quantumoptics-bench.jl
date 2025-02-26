using QuantumOptics
using BenchmarkTools

# global options
layout == "sparse" # "sparse" or "dense"

# parameters
num_modes = 12
num_tsave = 100
J = 1.0

# save times
gate_time = 1 / J
tspan = LinRange(0, gate_time, num_tsave)

# operators
b = SpinBasis(1//2)
sz = layout == "sparse" ? sigmaz(b) : dense(sigmaz(b))
id = layout == "sparse" ? identityoperator(b) : dense(identityoperator(b))
sigmazs = [tensor([j != m ? id : sz for j in 1:num_modes]...) for m in 1:num_modes]
identity = tensor([id for j in 1:num_modes]...)
H = J * sum([sigmazs[m] * sigmazs[m + 1] for m in 1:num_modes - 1])

# initial state
psi0 = tensor([spindown(b) for j in 1:num_modes]...)

# run benchmark
@benchmark timeevolution.schroedinger(tspan, psi0, H; abstol=1e-8, reltol=1e-6)
