The eigenvalues of the Hamiltonian determine almost all properties in a molecule or material of interest. The ground state for molecule Hamiltonian is of particular interest since the energy gap between the ground state and the first excited state of electrons at room temperature is usually larger. Most molecules are in the ground state.

Here, molecular electronic Hamiltonian is represented as \(\hat{H}\). A trial wave function \(\left| \varphi \left( \theta \right) \right\rangle\) is parameterized with \(\overrightarrow{\theta }\), which is called Ansatz. VQE is represented as follows:

\[\frac{\left\langle  \varphi \left( {\vec{\theta }} \right) \right|\hat{H}\left| \varphi \left( {\vec{\theta }} \right) \right\rangle }{\left\langle  \varphi \left( {\vec{\theta }} \right) \right|\left. \varphi \left( {\vec{\theta }} \right) \right\rangle }\ge {{E}_{0}}\]

\({{E}_{0}}\) is the lowest energy of molecular electronic Hamiltonian \(\hat{H}\). To estimate the \({{E}_{0}}\),  the left-hand side of the equation above is minimized by updating the parameters of the Ansatz \(\left| \varphi \left( \theta \right) \right\rangle\).

The molecular electronic Hamiltonian \(\hat{H}\) has the second quantized form:


\[\hat{H}=\sum\limits_{pq}{{{h}_{pq}}}\hat{a}_{p}^{+}{{\hat{a}}_{q}}+\frac{1}{2}\sum\limits_{pqrs}{{{h}_{pqrs}}}\hat{a}_{p}^{+}\hat{a}_{q}^{+}{{\hat{a}}_{r}}{{\hat{a}}_{s}}\]


It can be mapped to the linear combination of Pauli operator by Jordan-Wigner transformation or Bravyi-Kitaev transformation as follows:


\[\hat{H}=\sum\limits_{\alpha }{{{\omega }_{\alpha }}{{{\hat{P}}}_{\alpha }}}\]

Then, we can calculate the expectation of Hamitonian \(\hat{H}\) with respect to Ansatz \(\left| \varphi \left( \theta \right) \right\rangle\) by designing quantum circuit.  Finally, updating the parameters in the Ansatz \(\left| \varphi \left( \theta \right) \right\rangle\)  quantum circuit to get the \({{E}_{0}}\) the lowest energy of molecular electronic Hamiltonian.

The overview of VQE schematic diagram is as follows:

![](../figs/VQE-schematic-diagram.png)

VQE belongs to hybrid quantum-classical algorithms, in which quantum computer is responsible for executing quantum circuit and classical computer is responsible for updating the parameters of quantum gates in the Ansatz \(\left| \varphi \left( \theta \right) \right\rangle\). The lowest energy of molecular electronic Hamiltonian \(\hat{H}\)  can be obtained and the wave function corresponding to \({{E}_{0}}\).

To show the algorithm flow, we take the solution for molecule hydrogen's lowest energy as an example in the following code. After Jordan-Wigner transformation in a minimal basis molecule hydrogen Hamiltonian is

\[\begin{align}& {{{\hat{H}}}_{JW}}=-0.81261I+0.171201\sigma _{0}^{z}+0.171201\sigma _{1}^{z}-0.2227965\sigma _{2}^{z}-0.2227965\sigma _{3}^{z} \\ & +0.16862325\sigma _{1}^{z}\sigma _{0}^{z}+0.12054625\sigma _{2}^{z}\sigma _{0}^{z}+0.165868\sigma _{2}^{z}\sigma _{1}^{z}+0.165868\sigma _{3}^{z}\sigma _{0}^{z} \\ & +0.12054625\sigma _{3}^{z}\sigma _{1}^{z}+0.17434925\sigma _{3}^{z}\sigma _{2}^{z}-0.04532175\sigma _{3}^{x}\sigma _{2}^{x}\sigma _{1}^{y}\sigma _{0}^{y} \\& +0.04532175\sigma _{3}^{x}\sigma _{2}^{y}\sigma _{1}^{y}\sigma _{0}^{x}+0.04532175\sigma _{3}^{y}\sigma _{2}^{x}\sigma _{1}^{x}\sigma _{0}^{y}-0.04532175\sigma _{3}^{y}\sigma _{2}^{y}\sigma _{1}^{x}\sigma _{0}^{x}\\ \end{align}\]

We used two qubit in this example，the Ansatz \(\left| \varphi \left( \theta \right) \right\rangle\) is:

![](../figs/Ansatz.png)

Coefficients of the `H2` Hamiltonian.  
Hamiltonian gates: \(Z_0\), \(Z_1\), \(Z_0Z_1\), \(Y_0Y_1\), \(X_0X_1\).  
For more information, see [2](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.6.031007).

```python
coeffs = [-0.4804, +0.3435, -0.4347, +0.5716, +0.0910, +0.0910]
gates_group =  [
    ((0, "Z"),),
    ((1, "Z"),),
    ((0, "Z"), (1, "Z")),
    ((0, "Y"), (1, "Y")),
    ((0, "X"), (1, "X")),
]
```

`h2.isq` file.

```c++
import std;

int num_qubits = 2; 
int pauli_gates[] = [
    3, 0,
    0, 3,
    3, 3,
    2, 2,
    1, 1
];
// Hamiltonian gates: Z0, Z1, Z0Z1, Y0Y1, X0X2 

unit vqe_measure(qbit q[], int idx) {
    // using arrays for pauli measurement
    // I:0, X:1, Y:2, Z:3 
    int start_idx = num_qubits*idx;
    int end_idx = num_qubits*(idx+1);

    for i in start_idx:end_idx {
        if (pauli_gates[i] == 0) {
            // I:0
            continue;
        }
        if (pauli_gates[i] == 1) {
            // X:1
            H(q[i%num_qubits]);
            M(q[i%num_qubits]);
        }
        if (pauli_gates[i] == 2) {
            // Y:2
            X2P(q[i%num_qubits]);
            M(q[i%num_qubits]);
        }
        if (pauli_gates[i] == 3) {
            // Z:3
            M(q[i%num_qubits]);
        }
    }
}

unit main(int i_par[], double d_par[]) {

    qbit q[2];
    X(q[1]);

    Ry(1.57, q[0]);
    Rx(4.71, q[1]);
    CNOT(q[0],q[1]);
    Rz(d_par[0], q[1]);
    CNOT(q[0],q[1]);
    Ry(4.71, q[0]);
    Rx(1.57, q[1]);

    vqe_measure(q, i_par[0]);
}
```

Using [isqtools](https://www.arclightquantum.com/isqtools/basic/) to complie isq file.

```python
from isqtools import compile, simulate
compile("h2.isq", target="qir")
# compile "h2.isq" and generate the qir file "h2.so"
```

Define hyperparameters.


```python
import numpy as np 

shots = 100
learning_rate = 1.
energy_list = []
epochs = 20
theta = np.array([0.0])
```

Define functions.

```python
def get_expectation(theta):
    measure_results = np.zeros(len(gates_group) + 1)
    measure_results[0] = 1.0
    # The first does not require quantum measurement, which is constant
    # As a result, the other 5 coefficients need to be measured
    for idx in range(len(gates_group)):
        result_dict = simulate(
            "vqe2.so",
            shots=shots,
            int_param=idx,
            double_param=theta,
        )
        result = 0
        for res_index, frequency in result_dict.items():
            parity = (-1) ** (res_index.count("1") % 2)
            # parity checking to get expectation value
            result += parity * frequency / shots
            # frequency instead of probability
        measure_results[idx + 1] = result
        # to accumulate every expectation result
        # The result is multiplied by coefficient
    return np.dot(measure_results, coeffs)

def parameter_shift(theta):
    # to get gradients
    theta = theta.copy()
    theta += np.pi / 2
    forwards = get_expectation(theta)
    theta -= np.pi
    backwards = get_expectation(theta)
    return 0.5 * (forwards - backwards)
```

Run VQE.

```python
import time

energy = get_expectation(theta)
energy_list.append(energy)
print(f"Initial VQE Energy: {energy_list[0]} Hartree")

start_time = time.time()
for epoch in range(epochs):
    theta -= learning_rate * parameter_shift(theta)
    energy = get_expectation(theta)
    energy_list.append(energy)
    print(f"Epoch {epoch+1}: {energy} Hartree")

print(f"Final VQE Energy: {energy_list[-1]} Hartree")
print("Time:", time.time() - start_time)
```

Execution results:

    Initial VQE Energy: -0.28835999999999995 Hartree
    Epoch 1: -0.33809399999999995 Hartree
    Epoch 2: -0.444432 Hartree
    Epoch 3: -0.772314 Hartree
    Epoch 4: -1.387832 Hartree
    Epoch 5: -1.8272059999999999 Hartree
    Epoch 6: -1.824372 Hartree
    Epoch 7: -1.8609519999999997 Hartree
    Epoch 8: -1.8728760000000002 Hartree
    Epoch 9: -1.8528600000000002 Hartree
    Epoch 10: -1.842748 Hartree
    Epoch 11: -1.858316 Hartree
    Epoch 12: -1.847396 Hartree
    Epoch 13: -1.8391119999999999 Hartree
    Epoch 14: -1.855492 Hartree
    Epoch 15: -1.8314179999999998 Hartree
    Epoch 16: -1.8660059999999998 Hartree
    Epoch 17: -1.853266 Hartree
    Epoch 18: -1.87206 Hartree
    Epoch 19: -1.8629600000000002 Hartree
    Epoch 20: -1.8583160000000003 Hartree
    Final VQE Energy: -1.8583160000000003 Hartree
    Time: 14.92269778251648



**Reference**

1. J. Tilly, H. Chen, S. Cao, et al. "The variational quantum eigensolver: a review of methods and best practices." *Physics Reports*, 2022, 986: 1-128.
2. P. O’Malley et al. "Scalable Quantum Simulation of Molecular Energies." *Physics Review X*, 6, 031007, 2016.
