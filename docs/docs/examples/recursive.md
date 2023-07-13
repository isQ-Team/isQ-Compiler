Recursive Fourier sampling
--------------------------

To present the recursive Fourier sampling (RFS) problem, we begin by defining a type of tree. Let \(n\) and \(l\) be positive integers and consider a symmetric tree where each node, except the leaves, has \(2^n\) children, and the depth is \(l\). Let the root be labeled by \(\varnothing\). The root’s children are labeled \((x_1)\) with \(x_1\in\{0,1\}^n\). Each child of \((x_1)\) is, in turn, labeled \((x_1, x_2)\) with \(x_2\in\{0,1\}^n\). We continue until we have reached the leaves, which are labeled by \((x_1,\cdots,x_l)\). Thus each node’s label can be thought of as a path describing how to find the node from the root.

Now we add the Fourier component to a tree. We begin by fixing an efficiently computable function \(g:\{0,1\}^n\rightarrow\{0,1\}\). With each node of the tree \((x_1,\cdots,x_k)\), we associate a "secret" string \(s_{(x_1,\cdots,x_k)}\in\{0,1\}^n\). These secrets are promised to obey
\[g\left(s_{(x_1,\cdots,x_k)}\right)=s_{(x_1,\cdots,x_{k-1})}\cdot x_k\]
and the inner product has taken modulo 2. If \(k=1\), we take \(s_{(x_1,\cdots,x_{k-1})}\) to mean \(s_{(\varnothing)}\). In this way, each node’s secret encodes one bit of information about its parent’s secret.

Now suppose that we are given an oracle \(A:\left(\{0,1\}^n\right)^l\rightarrow\{0,1\}\) which behaves as
\[A(x_1,\cdots,x_l)=g\left(s_{(x_1,\cdots,x_l)}\right)\]
Note that \(A\) works for the leaves of the tree only. Our goal is to find \(g\left(s_{(\varnothing)}\right)\). This is the recursive Fourier sampling problem.

The recursive nature of the RFS problem is obvious. First, note that the subtree rooted at any node obeys the same promises as the whole tree. Thus each subtree defines an RFS problem. (The subtree rooted at a leaf is a trivial problem, where the oracle just returns the solution.) Thus we have a methodology for solving RFS problems: solve subtrees in order to determine information about the root’s secret, then calculate the secret. Solving subtrees uses the same method, except when we reach a leaf, where the oracle is used instead. This is a type of top-down recursive structure, where the tree is built from a number of subtrees with the same structure.

Before presenting the quantum solution, we first define the behavior of oracles \(A\) and \(G\):
\[A|x_1\rangle\cdots|x_l\rangle|y\rangle=|x_1\rangle\cdots|x_l\rangle|y\oplus g\left(s_{(x_1,\cdots,x_l)}\right)\rangle\]
\[G|s\rangle|y\rangle=|s\rangle|y\oplus g(s)\rangle\]
The main idea behind the algorithm is to use the fact that \(H^{\otimes n}\) transforms \(|y\rangle\) into \(\sum_s(-1)^{x\cdot y}|x\rangle\) and vice versa. We have phase feedback and a call to \(A\) to create the state \(\sum_{x_l}^{s_{(x_1,\cdots,x_{l-1})}\cdot x_l}|x\rangle\), then apply \(H^{\otimes n}\) to obtain \(|s_{(x_1,\cdots,x_{l-1})}\rangle\). After calculating \(g\left(s_{(x_1,\cdots,x_l)}\right)\), we uncompute \(|s_{(x_1,\cdots,x_{l-1})}\rangle\) to disentangle this register from others.



The process of algorithm QRFS is as follows:

&emsp;&emsp;Input: oracle \(A\) and (G\), \(l, k\), quantum registers \(x_1,...,x_k, y\)

​	&emsp;&emsp; 1. If \(k = l\) then apply \(A\) to \((x_1 ...x_l, y)\), then return.

​	&emsp;&emsp;2. Introduce ancilla \(x_{k+1}\) in the state \(\frac{1}{\sqrt{2^n}}\sum_{x \in\{0,1\}^n}|x>\)

​	&emsp;&emsp;3. Introduce ancilla \(y'\) in state  \(\frac{1}{\sqrt{2}}(|0>-|1>)\)

​	&emsp;&emsp;4. Call \(QRFS(A,G,l,k+1,(x_1...x_{k+1}),y′)\)

​	&emsp;&emsp;5. Apply \(H^{⊗n}\) on register \(x_{k+1}\).

​	&emsp;&emsp;6. Apply \(G\) to \((x_{k+1}, y)\).

​	&emsp;&emsp;7. Apply \(H^{⊗n}\) on register \(x_{k+1}\).

​	&emsp;&emsp;8. Call \(QRFS(A,G,l,k+1,(x_1...x_{k+1}),y′)\)

​	&emsp;&emsp;9. Discard \(x_{k+1}\) and \(y'\).

Now, we use isQ to compose an example of solving RFS. We set \(n=l=2\). So there are two labels \(x_1, x_2\), each one is 2 bits. \(A\) is a unitary gate acting on 5 bits, and \(G\) acts on 3 bits. Implement the function `recursive_fourier_sampling` according to the above algorithm process. 

In this example, according to the previous formula, we can calculate \(s_{00} = 11, s_{01} = 00, s_{10} = 10, s_{11} = 01 => s_{(\varnothing)} = 10 =>g\left(s_{(\varnothing)}\right) = 1 \).  So the measurement result of \(p[0]\) should be 1

```C++
import std;

oracle A(4,1) = [0,1,1,0,0,0,0,0,0,0,1,1,0,1,0,1];
oracle g(2,1) = [0,1,1,0]; 

int a;
qbit q[4], p[3];

procedure hadamard_layer(int k){
    H(q[2*k]);
    H(q[2*k+1]);
}

procedure recursive_fourier_sampling(int k){
	if (k == 2){
		A(q[0], q[1], q[2], q[3], p[2]);
	}else{
		hadamard_layer(k);
		X(p[k+1]);
		H(p[k+1]);

		recursive_fourier_sampling(k+1);

		hadamard_layer(k);
		g(q[2*k],q[2*k+1], p[k]);
		hadamard_layer(k);

		recursive_fourier_sampling(k+1);

		hadamard_layer(k);
		H(p[k+1]);
		X(p[k+1]);
	}
}

procedure main(){
	a = 0;
	recursive_fourier_sampling(a);
	print M(p[0]);
}
```

**Reference**

1. McKague, Matthew. "Interactive proofs with efficient quantum prover for recursive Fourier sampling." *arXiv preprint arXiv:1012.5699*, 2010.

2. Ethan Bernstein and Umesh Vazirani. Quantum complexity theory. *SIAM Journal on Computing*, 26(5):1411–1473, 1997.
