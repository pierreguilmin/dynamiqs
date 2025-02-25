# Python API

The **Dynamiqs** Python API features two main types of functions: solvers of differential equations describing quantum systems, and various utility functions to ease the creation and manipulation of quantum states and operators.

## Solvers

### General

::: dynamiqs.integrators
    options:
        table: true
        members:
        - sesolve
        - mesolve
        - sepropagator
        - mepropagator
        - floquet

### Stochastic

::: dynamiqs.integrators
    options:
        table: true
        members:
        - jssesolve
        - dssesolve
        - jsmesolve
        - dsmesolve

## Core

### Quantum arrays

::: dynamiqs.qarrays.qarray
    options:
        table: true
        members:
        - QArray

### Time-dependent qarrays

::: dynamiqs.time_qarray
    options:
        table: true
        members:
        - TimeQArray
        - constant
        - pwc
        - modulated
        - timecallable

### Methods (dq.method)

::: dynamiqs.method
    options:
        table: true
        members:
        - Tsit5
        - Dopri5
        - Dopri8
        - Kvaerno3
        - Kvaerno5
        - Euler
        - EulerMaruyama
        - Rouchon1
        - Expm
        - Event

### Gradients (dq.gradient)

::: dynamiqs.gradient
    options:
        table: true
        members:
        - Autograd
        - CheckpointAutograd

## Utilities

### Operators

::: dynamiqs.utils.operators
    options:
        table: true
        members:
        - eye
        - eye_like
        - zeros
        - zeros_like
        - destroy
        - create
        - number
        - parity
        - displace
        - squeeze
        - quadrature
        - position
        - momentum
        - sigmax
        - sigmay
        - sigmaz
        - sigmap
        - sigmam
        - hadamard
        - rx
        - ry
        - rz
        - sgate
        - tgate
        - cnot
        - toffoli


### States

::: dynamiqs.utils.states
    options:
        table: true
        members:
        - fock
        - fock_dm
        - basis
        - basis_dm
        - coherent
        - coherent_dm
        - ground
        - excited


### Quantum utilities

::: dynamiqs.utils
    options:
        table: true
        members:
        - dag
        - powm
        - expm
        - cosm
        - sinm
        - signm
        - trace
        - tracemm
        - ptrace
        - tensor
        - expect
        - norm
        - unit
        - dissipator
        - lindbladian
        - isket
        - isbra
        - isdm
        - isop
        - isherm
        - toket
        - tobra
        - todm
        - proj
        - braket
        - overlap
        - fidelity
        - purity
        - entropy_vn
        - bloch_coordinates
        - wigner
        namespace: utils/general/


### QArray utilities

::: dynamiqs.qarrays.utils
    options:
        table: true
        members:
        - asqarray
        - isqarraylike
        - stack
        - to_jax
        - to_numpy
        - to_qutip
        - sparsedia_from_dict

### Global settings

::: dynamiqs.utils.global_settings
    options:
        table: true
        members:
        - set_device
        - set_precision
        - set_matmul_precision
        - set_layout


### Vectorization

::: dynamiqs.utils.vectorization
    options:
        table: true
        members:
        - operator_to_vector
        - vector_to_operator
        - spre
        - spost
        - sprepost
        - sdissipator
        - slindbladian


### Quantum optimal control

::: dynamiqs.utils.optimal_control
    options:
        table: true
        members:
        - snap_gate
        - cd_gate


### Random (dq.random)

::: dynamiqs.random
    options:
        table: true
        members:
        - real
        - complex
        - herm
        - psd
        - dm
        - ket


### Plotting (dq.plot)

::: dynamiqs.plot
    options:
        table: true
        members:
        - wigner
        - wigner_mosaic
        - wigner_gif
        - pwc_pulse
        - fock
        - fock_evolution
        - hinton
        - xyz
        - gifit
        - grid
        - mplstyle

### Magic helpers

::: dynamiqs.hermitian_conjugate
    options:
        table: true
        members:
        - hc
