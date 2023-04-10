isQ examples
================

Now we will show you how to write isQ code with some examples

* [Bell State](#bell)

* [Bernstein Vazirani](#bv)

* [Grover Search](#gs)

* [Recursive Fourier Sampling](#rfs)

* [Variational Quantum Algorithm](#vqe)

</br>
<h2 id = "bell"></h2>

Bell State
-----------------

```C++
import std;

qbit a, b;

procedure main(){
    H(a);
    CNOT(a, b);
    int x = M(a);
    int y = M(b);
    print x;
    print y;
}

```

</br>
<h2 id = "bv"></h2>

Bernstein Vazirani
--------------------

```C++
import std;

// support 3-bit s is 110
oracle g(3, 1) = [0, 0, 1, 1, 1, 1, 0, 0];
qbit q[3], anc;

procedure main(){
    for i in 0:3{
        H(q[i]);
    }
    X(anc);
    H(anc);

    g(q[0], q[1], q[2], anc);
    for i in 0:3{
        H(q[i]);
        int a = M(q[i]);
        print a;
    }
}

```
</br>
<h2 id = "gs"></h2>

Grover Search
----------------

```C++
import std;

defgate U = [
    1, 0, 0, 0, 0, 0, 0, 0;
    0, -1, 0, 0, 0, 0, 0, 0;
    0, 0, -1, 0, 0, 0, 0, 0;
    0, 0, 0, -1, 0, 0, 0, 0;
    0, 0, 0, 0, -1, 0, 0, 0;
    0, 0, 0, 0, 0, -1, 0, 0;
    0, 0, 0, 0, 0, 0, -1, 0;
    0, 0, 0, 0, 0, 0, 0, -1
];

oracle G(3,1) = [0,1,0,0,0,0,0,0];

qbit q[3];
qbit anc;

procedure hardmard(){
    H(q[0]);
    H(q[1]);
    H(q[2]);
}

procedure init(){
    hardmard();
    X(anc);
    H(anc);
}

procedure grover_search(){
    G(q[0], q[1], q[2], anc);
    hardmard();
    U(q[0], q[1], q[2]);
    hardmard();
}

procedure main(){
    init();
    int a = 2;
    while a > 0{
        grover_search();
        a = a - 1;
    }
    H(anc);

    for i in 0:3{
        int x = M(q[i]);
        print x;
    }
}

```

</br>
<h2 id = "rfs"></h2>

Recursive Fourier Sampling
--------------------------

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
	int g = M<p[0]>;	
	print g;
}

```

</br>
<h2 id = "vqe"></h2>

Variational Quantum Algorithm
--------------------------

isQ implements variational quantum algorithm through python. Take the ground state energy of hydrogen molecule as an example, we can first write circuits with parameters through isQ

```c++
import std;

procedure main(int i_par[], double d_par[]){
    qbit q[2];
    X(q[1]);

    Ry(1.57, q[0]);
    Rx(4.71, q[1]);
    CNOT(q[0],q[1]);
    Rz(d_par[0], q[1]);
    CNOT(q[0],q[1]);
    Ry(4.71, q[0]);
    Rx(1.57, q[1]);

    if(i_par[0] == 0){
        M(q[0]);
    }
    if(i_par[0]==1){
        M(q[1]);
    }
    if(i_par[0]==2){
        M(q[0]);
        M(q[1]);
    }
    if(i_par[0]==3){
        Rx(1.57, q[0]);
        Rx(1.57, q[1]);
        M(q[0]);
        M(q[1]);
    }
    if(i_par[0]==4){
        H(q[0]);
        H(q[1]);
        M(q[0]);
        M(q[1]);
    }

}
```

Then, update parameters through python. In python, the compilation and simulation of isQ can be called through `os.popen`

```python
# ground state energy of hydrogen molecule
from numpy.random import rand
from scipy.optimize import minimize
import os
import json

# compile "h2.isq" and generate the qir file "h2.so"
h2_isq = 'h2.isq'
compile_cmd = f"isqc compile {h2_isq}"
res = os.popen(compile_cmd).read()
if res:
    print('compile error')
    print(res)
else:
    print('compile ok!')


h2_sim_file = 'h2.so'

def get_exception(theta) -> float:  
    '''
    theta: Angle during preparation
    e_n: E_N
    
    get the expectation value of 
    the Hamiltonian for specific theta
    '''
    theta = float(theta) # float
    
    hs = [-0.4804, +0.3435, -0.4347, +0.5716, +0.0910, +0.0910]
    # coefficients of the Hamiltonian
    # for more information, see `PHYS. REV. X 6, 031007 (2016)`
    
    exceptions = list() # to store results in a List
    
    exceptions.append(hs[0])
    E_N = 5 
    # The first does not require quantum measurement, which is constant
    # As a result, the other 5 coefficients need to be measured
    # i.e. hs[1], hs[2], hs[3], hs[4], hs[5]

    for e_n in range(E_N):

        # simulate and get the result
        simulate_cmd = f'isqc simulate -i {e_n} -d {theta} --shots 100 {h2_sim_file}'
        
        res = os.popen(simulate_cmd).read()
        
        try:
            
            test_res = json.loads(res)
            exception = 0 # initialize to 0
            
            for measure_res in test_res: # test_res is Dict
                
                frequency = test_res[measure_res]/100 # to get every frequency
                
                # 频率代替概率
                
                parity = (-1)**(measure_res.count('1') % 2) # to get every parity
                
                # 奇偶校验
                
                exception += parity * frequency # to accumulate every exception result
                
            exceptions.append(hs[e_n+1]*exception) # The result is multiplied by coefficients
        except:
            print('simulate error') # error
            print(res)
            exit()
            
    return sum(exceptions) # to get the final result of Hamiltonian with `hs`


# nelder-mead optimization of a convex function
# define range for theta

theta_min, theta_max = -3.14, 3.14

# define the starting point as a random sample from the domain
pt = theta_min + rand(1) * (theta_max - theta_min)

# perform the search

result = minimize(get_exception, pt, method='nelder-mead')

# summarize the result
print(f"Status : {result['message']}")
print(f"Total Evaluations: {result['nfev']}")


# evaluate solution
solution = result['x']
evaluation = get_exception(solution)
print(f"Solution: H_2({solution}) = {evaluation} Eh")
```