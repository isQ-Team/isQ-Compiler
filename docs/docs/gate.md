isQ supports the following quantum gates.

### X
```C++
X(qbit q)
```
Pauli X gate.
\[X=\begin{bmatrix}
0 & 1\\
1 & 0
\end{bmatrix}\]

### Y
```C++
Y(qbit q)
```
Pauli Y gate.
\[Y=\begin{bmatrix}
0 & -i\\
i & 0
\end{bmatrix}\]

### Z
```C++
Z(qbit q)
```
Pauli Z gate.
\[Z=\begin{bmatrix}
1 & 0\\
0 & -1
\end{bmatrix}\]

### H
```C++
H(qbit q)
```
Hardmard gate.
\[H=\frac{1}{\sqrt{2}}\begin{bmatrix}
1 & 1\\
1 & -1
\end{bmatrix}\]

### S
```C++
S(qbit q)
```
Phase gate.
\[S=\begin{bmatrix}
1 & 0\\
0 & i
\end{bmatrix}\]
Note that \(S^2=Z\).

### T
```C++
T(qbit q)
```
(\pi/8) gate.
\[T=\begin{bmatrix}
1 & 0\\
0 & e^{i\pi/4}
\end{bmatrix}\]
Note that \(T^2=S\).

### Rx
```C++
Rx(double theta, qbit q)
```
Rotation about the X axis.
\[Rx(\theta)=e^{-i\theta X/2}=\begin{bmatrix}
\textrm{cos}\frac{\theta}{2} & -i\textrm{sin}\frac{\theta}{2}\\
-i\textrm{sin}\frac{\theta}{2} & \textrm{cos}\frac{\theta}{2}
\end{bmatrix}\]

### Ry
```C++
Ry(double theta, qbit q)
```
Rotation about the Y axis.
\[Ry(\theta)=e^{-i\theta Y/2}=\begin{bmatrix}
\textrm{cos}\frac{\theta}{2} & -\textrm{sin}\frac{\theta}{2}\\
\textrm{sin}\frac{\theta}{2} & \textrm{cos}\frac{\theta}{2}
\end{bmatrix}\]

### Rz
```C++
Rz(double theta, qbit q)
```
Rotation about the Z axis.
\[Rz(\theta)=e^{-i\theta Z/2}=\begin{bmatrix}
e^{-i\theta/2} & 0\\
0 & e^{i\theta/2}
\end{bmatrix}\]

### X2P
```C++
X2P(qbit q)
```
\(\pi/2\) rotation about the X axis.
\[X2P=Rx(\pi/2)=\frac{1}{\sqrt{2}}\begin{bmatrix}
1 & -i\\
-i & 1
\end{bmatrix}\]

### X2M
```C++
X2M(qbit q)
```
\(-\pi/2\) rotation about the X axis.
\[X2M=Rx(-\pi/2)=\frac{1}{\sqrt{2}}\begin{bmatrix}
1 & i\\
i & 1
\end{bmatrix}\]

### Y2P
```C++
Y2P(qbit q)
```
\(\pi/2\) rotation about the Y axis.
\[Y2P=Rx(\pi/2)=\frac{1}{\sqrt{2}}\begin{bmatrix}
1 & -1\\
1 & 1
\end{bmatrix}\]

### Y2M
```C++
Y2M(qbit q)
```
\(-\pi/2\) rotation about the Y axis.
\[Y2M=Rx(-\pi/2)=\frac{1}{\sqrt{2}}\begin{bmatrix}
1 & 1\\
-1 & 1
\end{bmatrix}\]

### U3
```C++
U3(double theta, double phi, double lambda, qbit q)
```
Generic single-qubit rotation gate with 3 Euler angles.
\[U3(\theta,\phi,\lambda)=\begin{bmatrix}
\textrm{cos}\frac{\theta}{2} & -e^{i\lambda}\textrm{sin}\frac{\theta}{2}\\
e^{i\phi}\textrm{sin}\frac{\theta}{2} & e^{i(\phi+\lambda)}\textrm{cos}\frac{\theta}{2}
\end{bmatrix}\]
Note that
\[U3\left(\theta,-\frac{\pi}{2},\frac{\pi}{2}\right)=Rx(\theta)\]
\[U3(\theta,0,0)=Ry(\theta)\]

### CNOT
```C++
CNOT(qbit c, qbit t)
```
Controlled-NOT gate.
\[CNOT=\begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 0 & 1\\
0 & 0 & 1 & 0
\end{bmatrix}\]

### CZ
```C++
CZ(qbit c, qbit t)
```
Controlled-Z gate.
\[CZ=\begin{bmatrix}
1 & 0 & 0 & 0\\
0 & 1 & 0 & 0\\
0 & 0 & 1 & 0\\
0 & 0 & 0 & -1
\end{bmatrix}\]

### Toffoli
```C++
Toffoli(qbit c1, qbit c2, qbit t)
```
Toffoli gate.
\[Toffoli=\begin{bmatrix}
1 & 0 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 1 & 0 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 1 & 0 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 1 & 0 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 1 & 0 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 1 & 0 & 0\\
0 & 0 & 0 & 0 & 0 & 0 & 0 & 1\\
0 & 0 & 0 & 0 & 0 & 0 & 1 & 0
\end{bmatrix}\]

### GPhase
```C++
GPhase(double theta)
```
Global phase gate.
\[GPhase(\theta)=e^{i\theta}\]
Note that this gate does not have any `qbit`-type parameter.
