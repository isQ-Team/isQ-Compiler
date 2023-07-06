<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML"></script>
<script type="text/x-mathjax-config">
    MathJax.Hub.Config({ tex2jax: {inlineMath: [["\(", "\)"]], displayMath: [["\[", "\]"]]}, messageStyle: "none" });
</script>

Grover's algorithm
----------------
Grover's algorithm finds with high probability the unique input to a black box function that produces a particular output value, using just \(O(\sqrt {N})\) evaluations of the function, where \(N\) is the size of the function's domain. As a comparison, the analogous problem in classical computation cannot be solved in fewer than \(O(N)\) evaluations. Although Grover's algorithm provides only a quadratic speedup, it is considerable when \(N\) is large, and Grover's algorithm can be applied to speed up broad classes of algorithms.

As input for Grover's algorithm, suppose we have a function \(f:\{0,1,\cdots,N-1\}\rightarrow\{0,1\}\). In the "unstructured database" analogy, the domain represents indices to a database, and \(f(x) = 1\) if and only if the data that \(x\) points to satisfies the search criterion. We additionally assume that only one index satisfies \(f(x) = 1\), and we call this index \(\omega\). Our goal is to identify \(\omega\).

We can access \(f\) with an oracle in the form of a unitary operator \(U_w\) that acts as
\[U_\omega|x\rangle=(-1)^{f(x)}|x\rangle\]
i.e.,
\[\left\{\begin{array}{l}
U_\omega|x\rangle=-|x\rangle\quad\text{for }x=\omega,\\
U_\omega|x\rangle=|x\rangle\quad\ \ \text{for }x\neq\omega
\end{array}\right.\]
This uses the \(N\)-dimensional state space \(\mathcal{H}\), which is supplied by a register with \(n=\lceil\text{log}_2N\rceil\) qubits.

The steps of Grover's algorithm are

1. Initialize the system to uniform superposition over all states
\[|s\rangle=\frac{1}{\sqrt{N}}\sum_{x=0}^{N-1}|x\rangle\]

2. Perform the following "Grover iteration" \(O(\sqrt{N})\) times:

    1. Apply oracle \(U_\omega\);

    2. Apply the Grover diffusion operator \(U_s=2|s\rangle\langle s|-I\)

3. Measure the resulting quantum state in the computational basis.

The circuit is shown as the following:

<img src="../../figs/grover.svg" style="zoom:150%;" />

We write an example using isQ to demonstrate Grover's algorithm. In this example, \(N=8\), corresponding to the Hilbert space of 3 qubits. With this size, we can calculate that 2 Grover iterations are needed. We select \(\omega=1\), i.e., only \(f(1)=1\). This oracle is defined using a truth table. Next, we define \(U0=2|0\rangle\langle0|-I\) using a unitary matrix. The full program is shown below.

```C++
import std;

oracle U_omega(3,1) = [0,1,0,0,0,0,0,0];

defgate U0 = [
    1, 0, 0, 0, 0, 0, 0, 0;
    0, -1, 0, 0, 0, 0, 0, 0;
    0, 0, -1, 0, 0, 0, 0, 0;
    0, 0, 0, -1, 0, 0, 0, 0;
    0, 0, 0, 0, -1, 0, 0, 0;
    0, 0, 0, 0, 0, -1, 0, 0;
    0, 0, 0, 0, 0, 0, -1, 0;
    0, 0, 0, 0, 0, 0, 0, -1
];

qbit q[3];
qbit anc;

int grover_search() {
    // Initialize
    H(q);
    X(anc);
    H(anc);

    // Grover iterations
    for i in 0:2 {
        U_omega(q[2], q[1], q[0], anc);
        H(q);
        U0(q[2], q[1], q[0]);
        H(q);
    }

    // Finilize
    H(anc);
    X(anc);
    return M(q);
}

procedure main(){
    print grover_search();
}
```

We executed the program 100 times using isQ compiler's command `bin/isqc run --shots 100 examples/grover.isq`. The execution result is
```
{"010": 1, "110": 1, "011": 1, "001": 97}
```
Most of the results (97 out of 100) are 1, equal to \(\omega\). It shows that Grover's algorithm finds the desired answer with a high probability.
