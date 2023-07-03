import numpy as np
import math


def next2pow(n):
    return pow(2, math.ceil(math.log(n)/math.log(2)))


def generate_subspace_unitary(n, d, indices):
    extended = next2pow(n)
    #print("extended: "+str(extended))

    U = np.eye(extended, dtype = float)

    #print(indices)
    #print(d)

    for index in np.nditer(indices):
        for id in np.nditer(indices):
            U[index, id] = 2/d
            if index == id:
                U[index, id] -= 1
    
    return U


def print_isq_gate(U, name):

    arr = ""
    extended = U.shape[0]

    indent = "\t"
    
    for i in range(extended):
        line = indent
        for j in range(extended-1):
            line += str(U[i,j]) + ", "
        line += str(U[i, extended-1])
        if i != extended-1:
            line += ";"
        line += "\n"
        arr += line
    

    code = "defgate " + name + " = [\n" + arr + "];\n"
    return code

def gen_ctrl_index(i, steps):
    ctrl_str = ""
    while steps > 0:
        if i & 1 == 0:
            ctrl_str += "nctrl "
        else:
            ctrl_str += "ctrl "
        i >>= 1
        steps -= 1

    return ctrl_str

def generate_coin_operator(n, graph, nameA, nameB):
    
    code = ""
    gate_defs = ""
    coin_proc = ""
    indent = "\t"

    qubit_count = math.ceil(math.log(n)/math.log(2))
    
    for i in range(n):
        indices = np.nonzero(graph[i])
        # print(indices)

        d = np.count_nonzero(graph[i])
        U = generate_subspace_unitary(n, d, indices)
        gate_defs += print_isq_gate(U, nameA+str(i))
        gate_defs += "\n"

        params = ""
        for j in range(qubit_count):
            params += nameA + "[" + str(j) + "], "

        for j in range(qubit_count):
            params += nameB + "[" + str(qubit_count-1-j) + "]";
            if j != qubit_count-1:
                params += ", "

        coin_proc += indent + gen_ctrl_index(i, math.log(U.shape[0])) + nameA + str(i) + "(" + params + ");\n"

    coin_proc = "procedure Grover_coin(qbit "+nameA + "["+ str(qubit_count) + "], qbit " + nameB + "["+ str(qubit_count) +"]){\n" + coin_proc + "}\n"

    code = gate_defs + coin_proc + "\n"

        #print(gen_ctrl_index(i, math.log2(U.shape[0])))
        #print(s)
    
    return code

def gen_rev_arc_proc(qubit_count, nameA, nameB):
    
    indent = "\t"

    code = ""
    code += "procedure rev_arc(qbit "+nameA+"["+str(qubit_count)+"], qbit "+nameB+"["+str(qubit_count)+"]){\n"
    
    code += indent + "for i in 0:"+str(qubit_count) + "{\n"
    code += indent + indent + "SWAP("+nameA+"[i], "+nameB+"[i]);\n"
    code += indent + "}\n"

    code += "}\n"
    code += "\n"
    return code

def gen_walk_proc(qubit_count, nameA, nameB):
    
    indent = "\t"
    code = ""
    code += "procedure walk(qbit "+nameA+"["+str(qubit_count)+"], qbit "+nameB+"["+str(qubit_count)+"]){\n"
    
    code += indent + "Grover_coin("+nameA+", "+nameB+");\n"
    code += indent + "rev_arc("+nameA+", "+nameB+");\n"

    code += "}\n"
    code += "\n"
    
    return code


def gen_main(qubit_count, nameA, nameB, steps):
    indent = "\t"
    code = ""
    code += "procedure main(){\n"
    
    code += indent + "qbit "+nameA+"["+str(qubit_count)+"], "+nameB + "[" + str(qubit_count)+"];\n"
    code += indent + "for i in 0:" + str(steps) + "{\n"
    code += indent + indent + "walk("+nameA+", "+nameB+");\n"
    code += indent + "}\n"
    code += indent + "M("+nameA +");\n"

    code += "}\n"
    return code


def gen_isq_for_arcrev_walk(n, graph, nameA, nameB, steps):
    
    extended = next2pow(n)
    qubit_count = math.ceil(math.log2(extended))
    
    program = ""

    program += "import std;\n\n"

    program += generate_coin_operator(n, graph, nameA, nameB)
    program += gen_rev_arc_proc(qubit_count, nameA, nameB)
    program += gen_walk_proc(qubit_count, nameA, nameB)
    program += gen_main(qubit_count, nameA, nameB, steps)

    return program

n = 6
graph = np.array([[1, 1, 1, 0, 0, 0], [1, 1, 0, 1, 1, 0], [1, 0, 1, 1, 1, 0], [0, 1, 1, 1, 0, 1], [0, 1, 1, 0, 1, 1], [0, 0, 0, 1, 1, 1]], np.int32)

nameA = "U_"
nameB = "V_"

steps =3

print(gen_isq_for_arcrev_walk(n, graph, nameA, nameB, steps))
