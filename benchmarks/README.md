# Benchmarks

This directory contains a set of scripts to benchmark dynamiqs against other libraries (`QuTiP`, `QuTiP-jax` and `QuantumOptics.jl` for now), on a set of representative problems of the simulation of dynamics of closed and open quantum systems.

All scripts have the same structure:
- Import relevant dependencies
- Define global options, namely the matrix layout (dense or sparse) and the backend (CPU or GPU) when applicable
- Define the problem to solve (Hamiltonian, jump operators, initial state, time grid, etc.)
- Define the solver and its options
- Run the benchmark using `timeit` for Python or `BenchmarkTools.jl` for Julia

For fair comparisons, all scripts use the default solver of the library (namely `Tsit5` for dynamiqs, `QuTiP-jax` and `QuantumOptics.jl`, and `Adams` for `QuTiP`) with the same tolerances (`atol=1e-8`, `rtol=1e-6`).

## Selected problems

Below, we provide a complete description of each problem considered in these benchmarks. The following table gives a summary of the main characteristics of these problems.

| | Cross-resonance gate | Dissipative cat CNOT | Driven-dissipative Kerr oscillator | Transmon pi-pulse | 1D Ising model |
|----------------------|----------------------|----------------------|----------------------|----------------------|----------------------|
| Time-dependence | time-dependent | constant | constant | time-dependent | constant |
| Nature of system | closed | open | open | open | closed |
| Hilbert space size | 4 | 1024 | 32 | 3 | 4096 |
| Number of modes | 2 | 2 | 1 | 1 | 12 |
| Batching size | 1 | 1 | 20 | 400 | 1 |

### Cross-resonance gate

This problem simulates the dynamics of a cross-resonance gate between two qubits. The Hamiltonian is given by
$$
H = \omega_1 \sigma_z^{(1)} + \omega_2 \sigma_z^{(2)} + J (\sigma_+^{(1)} \sigma_-^{(2)} + \mathrm{h.c.}) + \epsilon(t) \sigma_x^{(1)},
$$
where $\epsilon(t) = \epsilon_0 \cos(\omega_d t)$ with $\omega_d = \omega_2 - J^2 / (\omega_1 - \omega_2)$. The system is simulated for a duration $T = \frac{\pi |\omega_2 - \omega_1|}{2 J \epsilon_0}$. The state is initialized in $|00\rangle$. The selected system parameters are $\omega_1 = 4$, $\omega_2 = 6$, $J = 0.4$, and $\epsilon_0 = 0.4$.

### Dissipative cat CNOT

This problem simulates the dynamics of a CNOT gate between two dissipative cat qubits, encoded in quantum harmonic oscillators. The Hamiltonian is given by
$$
H = g (a_c^\dagger + a_c) (a_t^\dagger a_t - |\alpha|^2)
$$
and the unique jump operator reads
$$
L = \sqrt{\kappa_2} (a_c^2 - \alpha^2)
$$
The system is simulated for a duration $T = \pi / 4 \alpha g$. The state is initialized in $|\mathcal{C}_\alpha^+ \rangle |\mathcal{C}_\alpha^+ \rangle$ where $|\mathcal{C}_\alpha^+ \rangle$ is the even cat state. The selected system parameters are $\kappa_2 = 1$, $g = 0.3$, $\alpha^2 = 4$, and $N = 32$ for each mode.

### Driven-dissipative Kerr oscillator

This problem simulates the dynamics of an idling driven-dissipative Kerr oscillator, where the drive strength is sweeped. The Hamiltonian is given by
$$
H = K a^{\dagger 2} a^2 + \epsilon (a + a^\dagger)
$$
and the unique jump operator reads
$$
L = \sqrt{\kappa} a
$$
The system is simulated for a duration $T = \pi / K$. The state is initialized in a coherent state $|\alpha\rangle$. The selected system parameters are $K = 1$, $\kappa = 0.1$, $\alpha = 2$, $N = 32$ and $\epsilon$ is batched from $0$ to $0.5$ over $n_{batch} = 20$ parameters.

### Transmon pi-pulse

This problem simulates the dynamics of a 3-level transmon qubit under a gaussian-shaped $\pi$-pulse, where two parameters of the drive are batched over. The Hamiltonian is given by
$$
H = K a^{\dagger 2} a^2 + \Omega(t) (a + a^\dagger)
$$
where
$$
\Omega(t) = \Omega_0 \left[\exp\left(-\frac{(t - T/2)^2}{2\sigma^2}\right) - \exp\left(-\frac{T^2}{8\sigma^2}\right) \right]^2
$$
The system is simulated for a fixed duration $T$. The state is initialized in the transmon ground state $|g\rangle$. The selected system parameters are $K = 2$, $\kappa = 0.1$, $T = 20$, $\Omega_0$ is batched from $0$ to $0.3$ over $n_{batch, 1} = 20$ parameters, and $\sigma$ is batched from $0.5$ to $10$ over $n_{batch, 2} = 20$ parameters.

### 1D Ising model

This problem simulates the dynamics of a 1D chain of 12 spins under the Ising model Hamiltonian. The Hamiltonian is given by
$$
H = -J \sum_{i} \sigma_z^{(i)} \sigma_z^{(i+1)}
$$
with $J < 0$ (anti-ferromagnetic). The system is simulated for a duration $T = 1 / J$. The state is initialized in the ground state of each individual spin, $\otimes_i \vert\!\downarrow\rangle^{(i)}$. The selected system parameters are $J = -1$ and $n_{mode} = 12$ spins.
