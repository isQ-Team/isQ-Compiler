# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 23:43:55 2021

@author: zhoux
"""

'''
1、qubit数目：n

2、芯片拓扑结构：n*n的01矩阵，在python里用2维的ndarray表示（numpy.array, type=int），例如：
[0 1 0
1 0 1
0 1 0]
表示qubit 0和qubit 1相邻，qubit 1和qubit 2相邻。只有相邻的qubits之间才能作用2-qubit gates

3、逻辑电路：用一个list表示，list里面每一个元素都是一个gate。gate用3元组（python里的tuple）来表示，用元组（A，B，C）举例：
  1) A等于1或2，表示单比特门或两比特门；
  2) B表示gate的名称：当A==1时，B='Rx'，'Ry'，'Rz'或'H'，'S'，'T'等基本门，当A==2时，B='CNOT'或'CZ'（两比特门在一个电路中通常只有一类，全为CNOT或全为CZ）；
  3) C表示gate作用的逻辑比特编号：当A==1时，C是0...n-1的整数；当A==2时，C是一个tuple——(q1,q2)，q1是控制比特，q2是受控比特
'''
import networkx as nx
import numpy as np

def array_to_ag(adjacency):
    '''
    Generate Architecture graph representing conntctivity constraints of a QPU
    '''
    # check input legality
    if (len(adjacency.shape) != 2):
        raise('The dimensions of the input adjacency matrix cannot be {}'\
              .format(len(adjacency.shape)))
    if (adjacency.shape[0] != adjacency.shape[0]):
        raise('The input adjacency matrix must be symmetric!')
    if (np.sum(np.abs(adjacency - adjacency.transpose())) > 0.00001):
        raise('The input adjacency matrix must be symmetric!')
    num_q = adjacency.shape[0]
    # Generate AG
    AG = nx.Graph()
    AG.add_nodes_from(range(num_q))
    for q0 in range(num_q):
        for q1 in range(num_q):
            if adjacency[q0][q1] == 1: AG.add_edge(q0, q1)
    # add edge_2_index
    AG.edge_2_index = {}
    i = -1
    for edge in AG.edges():
        i += 1
        AG.edge_2_index[edge] = i
    # add shortest length and path to AG
    AG.shortest_length = dict(nx.shortest_path_length(AG, source=None,
                                                           target=None,
                                                           weight=None,
                                                           method='dijkstra'))
    AG.shortest_path = nx.shortest_path(AG, source=None, target=None, 
                                             weight=None, method='dijkstra')
    return AG

def cir_list_to_dg(cir_list, num_q):
    '''
    DG is a Directed Graph representing a quantum circuit in which each node
    represents a two-qubit gate.
    Attribues of DG:
        single_gates_before_first_layer:
            the single-q gates before the first cx layer
    Attribues of a node n (DG.nodes[n]['attribue_name']):
        gate: entry in the circuit list
        operand: indices of the (two) operand qubits
        single_gates0: list of single-qubit gates right after the fisrt operand 
        qubit of this node
        single_gates1: list of single-qubit gates right after the second operand
        qubit of this node
    '''
    # a list in which each element is the node that takes the first place of 
    # qubits
    first_nodes = [-1] * num_q
    front_nodes = [-1] * num_q
    
    i = -1
    DG = nx.DiGraph()
    num_q_log = 0
    # number of single-qubit gate befor the first CX layer
    # each element corresponds to a logical qubit
    DG.single_gates_before_first_layer = []
    for gate in cir_list:
        i += 1
        qubits = gate[2]
        num_q_gate = gate[0]
        if num_q_gate == 1:
            qubit = qubits
            if qubit + 1 > num_q_log: num_q_log = qubit + 1
            if gate[1] == 'M': continue
            node = front_nodes[qubit]
            if node == -1:
                DG.single_gates_before_first_layer.append(gate)
            else:
                if qubit == DG.nodes[node]['operand'][0]:
                    DG.nodes[node]['single_gates0'].append(gate)
                else:
                    DG.nodes[node]['single_gates1'].append(gate)
        if num_q_gate == 2:
            edges = []
            for qubit in qubits:
                if first_nodes[qubit] == -1:
                    first_nodes[qubit] = i
                else:
                    # record edges
                    edges.append((front_nodes[qubit], i))
                if qubit + 1 > num_q_log: num_q_log = qubit + 1
                front_nodes[qubit] = i
            # add node and edges
            DG.add_node(i,
                        gate=gate,
                        operand=qubits,
                        single_gates0=[],
                        single_gates1=[])
            for edge in edges:
                DG.add_edge(edge[0], edge[1])
        if num_q_gate > 2:
            raise(Exception('Qubit number of a gate cannot be more than 2.'))
    DG.first_gates = first_nodes
    DG.num_q_log = num_q_log
    return DG

def qiskit_gate_name_convert(qiskit_gate_name):
    if qiskit_gate_name == 'cx':
        return 'CNOT'
    return qiskit_gate_name

def qiskit_gate_convert(qiskit_gate):
    '''convert a qiskit gate to a tuple in the circuit list'''
    gate_name = qiskit_gate[0].name
    qargs = qiskit_gate[1]
    num_q = len(qargs)
    para = qiskit_gate[0].params
    if num_q == 1:
        qubits = qargs[0].index
    else:
        qubits = []
        for q in qargs: qubits.append(q.index)
        qubits = tuple(qubits)
    return num_q, qiskit_gate_name_convert(gate_name), qubits, para

suppurted_gates = {'CNOT', 'SWAP', 'p', 'h', 'u2', 'u3'}

def gate_to_qiskit_cir(cir_qiskit, gate):
    '''Convert a gate and add it to a qiskit quantum circuit'''
    num_q, name, qubits, paras = gate
    if not name in suppurted_gates: raise()
    if name == 'CNOT': cir_qiskit.cx(qubits[0], qubits[1])
    if name == 'SWAP': cir_qiskit.swap(qubits[0], qubits[1])
    if name == 'p': cir_qiskit.p(paras[0], qubits)
    if name == 'h': cir_qiskit.h(paras[0], qubits)
    if name == 'u2': cir_qiskit.u2(paras[0], paras[1], qubits)
    if name == 'u3': cir_qiskit.u3(paras[0], paras[1], paras[2], qubits)
    
#def cir_list_to_qiskit_cir(cir_list, num_q):
#    '''Convert a circuit list to a qiskit quantum circuit object'''
#    from qiskit import QuantumRegister, QuantumCircuit
#    cir = QuantumCircuit(QuantumRegister(num_q))
#    for gate in cir_list: gate_to_qiskit_cir(cir, gate)
#    return cir

#def qasm_to_cir_list(file_path):
#    '''
#    Convert a qasm file to circuit list
#    Return:
#        circuit_list, # qubits
#    '''
#    from qiskit import QuantumCircuit
#    # import qasm file as qiskit circuit
#    QASM_file = open(file_path, 'r')
#    iter_f = iter(QASM_file)
#    QASM = ''
#    for line in iter_f:
#        # read file line by line
#        QASM = QASM + line
#    qiskit_cir = QuantumCircuit.from_qasm_str(QASM)
#    QASM_file.close()
#    # qiskit circuit to circuit list
#    cir_list = []
#    qregs = qiskit_cir.qregs
#    if len(qregs) > 1:
#        # currently we do not support qiskit circuit with more than 1 quantum
#        # register
#        raise Exception('Qiskit circuit has more than 1 quantum register!')
#    data = qiskit_cir.data
#    for qiskit_gate in data:
#        gate = qiskit_gate_convert(qiskit_gate)
#        cir_list.append(gate)
#    return cir_list, len(qiskit_cir)

def qasm_to_dg(file_path):
    cir_list, num_q = qasm_to_cir_list(file_path)
    return cir_list_to_dg(cir_list, num_q)

def convert_mappings(mapping1, mapping2):
    '''
    Generate SWAP gates to convert mapping1 (logical to physical) to 
    mapping2
    '''
    cir_list = []
    mapping1 = mapping1.copy()
    for q_log1 in range(len(mapping1)):
        q_phy1 = mapping1[q_log1]
        q_phy2 = mapping2[q_log1]
        if q_phy1 == q_phy2: continue
        cir_list.append((2, 'SWAP', (q_phy1, q_phy2), []))
        for i in range(len(mapping1)):
            if mapping1[i] == q_phy2:
                mapping1[i] = q_phy1
                break
        mapping1[q_log1] = q_phy2
    return cir_list

def equivalence_checker_qct(cir_list_log,
                            cir_list_phy, init_map_list, 
                            ag_matrix, swap_name='SWAP'):
    '''
    Check the equivalence between two circuits
    ATTENTION: this method is not 100% accurate, we will ignore the single-qubit
    gates before the first layer of two-qubit gates
    '''
    from front_circuit import FrontCircuit
    AG = array_to_ag(ag_matrix)
    num_q_phy = len(AG)
    DG_log = cir_list_to_dg(cir_list_log, num_q_phy)
    DG_phy = cir_list_to_dg(cir_list_phy, num_q_phy)
    cir_log, cir_phy = FrontCircuit(DG_log, AG), FrontCircuit(DG_phy, AG)
    # init mapping for physical circuit
    num_q_log = DG_log.num_q_log
    log_to_phy = [-1] * num_q_log
    phy_to_log = [-1] * num_q_phy
    for q_log in range(num_q_log):
        q_phy = init_map_list[q_log]
        log_to_phy[q_log] = q_phy
        phy_to_log[q_phy] = q_log
    # check eq. layer-by-layer
    while cir_phy.num_remain_gates > 0:
        # execute SWAP gates
        while True:
            count = 0
            for node_dg in cir_phy.front_layer:
                if DG_phy.nodes[node_dg]['gate'][1] == swap_name:
                    count += 1
                    # update mapping
                    q_phy0, q_phy1 = DG_phy.nodes[node_dg]['operand']
                    q_log0, q_log1 = phy_to_log[q_phy0], phy_to_log[q_phy1]
                    phy_to_log[q_phy0] = q_log1
                    phy_to_log[q_phy1] = q_log0
                    if q_log0 != -1: log_to_phy[q_log0] = q_phy1
                    if q_log1 != -1: log_to_phy[q_log1] = q_phy0
                    # execute swap
                    cir_phy.execute_gate(node_dg)
            if count == 0: break
        # check eq. of the first physical gate in the front layer
        node_dg_phy = cir_phy.front_layer[0]
        name_phy = DG_phy.nodes[node_dg_phy]['gate'][1]
        q0_phy, q1_phy = DG_phy.nodes[node_dg_phy]['operand']
        q0_log, q1_log = phy_to_log[q0_phy], phy_to_log[q1_phy]
        phy_single_gates0 = DG_phy.nodes[node_dg_phy]['single_gates0']
        phy_single_gates1 = DG_phy.nodes[node_dg_phy]['single_gates1']
        find_eq_gate = False
        # try to find eq. gate in logical circuit
        for node_dg_log in cir_log.front_layer:
            name_log = DG_log.nodes[node_dg_log]['gate'][1]
            if name_phy != name_log: continue
            q2_log, q3_log = DG_log.nodes[node_dg_log]['operand']
            log_single_gates0 = DG_log.nodes[node_dg_log]['single_gates0']
            log_single_gates1 = DG_log.nodes[node_dg_log]['single_gates1']
            if (q0_log, q1_log) == (q2_log, q3_log):
                find_eq_gate = True
                # check single-qubit gates
                if len(log_single_gates0) != len(phy_single_gates0):
                    find_eq_gate = False
                    break
                if len(log_single_gates1) != len(phy_single_gates1):
                    find_eq_gate = False
                    break
                for gate_log, gate_phy in zip(log_single_gates0,
                                              phy_single_gates0):
                    if gate_log[1] != gate_phy[1]:
                        find_eq_gate = False
                        break
                for gate_log, gate_phy in zip(log_single_gates1,
                                              phy_single_gates1):
                    if gate_log[1] != gate_phy[1]:
                        find_eq_gate = False
                        break
            if (q0_log, q1_log) == (q3_log, q2_log):
                find_eq_gate = True
                # check single-qubit gates
                if len(log_single_gates0) != len(phy_single_gates1):
                    find_eq_gate = False
                    break
                if len(log_single_gates1) != len(phy_single_gates0):
                    find_eq_gate = False
                    break
                for gate_log, gate_phy in zip(log_single_gates0,
                                              phy_single_gates1):
                    if gate_log[1] != gate_phy[1]:
                        find_eq_gate = False
                        break
                for gate_log, gate_phy in zip(log_single_gates1,
                                              phy_single_gates0):
                    if gate_log[1] != gate_phy[1]:
                        find_eq_gate = False
                        break
            if find_eq_gate == True: break
        if find_eq_gate == False:
            return False
        else:
            cir_log.execute_gate(node_dg_log)
            cir_phy.execute_gate(node_dg_phy)
    if cir_log.num_remain_gates > 0: return False
    return find_eq_gate


#def equivalence_checker_qiskit(cir_list_log,
#                               cir_list_phy, final_map_list, 
#                               ag_matrix, swap_name='SWAP'):
#    '''
#    Check the equivalence between two circuits.
#    This method invokes Qiskit to perform classical simulation and compare 
#    unitaries of the two circuits.
#    '''
#    from qiskit import BasicAer, execute
#    backend = BasicAer.get_backend('unitary_simulator')
#    num_q = ag_matrix.shape[0]
#    swap_list = convert_mappings(final_map_list, list(range(num_q)))
#    cir_list_phy.extend(swap_list)
#    cir1 = cir_list_to_qiskit_cir(cir_list_log, num_q)
#    cir2 = cir_list_to_qiskit_cir(cir_list_phy, num_q)
#    job1 = execute(cir1, backend)
#    job2 = execute(cir2, backend)
#    U1 = job1.result().get_unitary()
#    U2 = job2.result().get_unitary()
#    error = np.sum(np.abs(U1 - U2))
#    if error < 0.0000001:
#        return True
#    else:
#        return False
    

def cir_list_attributes(cir_list, gate_depth={},
                        error_matrix=[], flag_print=False):
    '''Calculate size, depth and success rate (if error matrix is given)'''
    # get size
    size = 0
    for num_q, name, qubits, _ in cir_list:
        size_add = gate_depth[name] if name in gate_depth else 1
        size += size_add
    # get depth
    ## by default, if the gate type is not in gate_depth, it takes 1 depth
    qubit_to_depth = []
    for num_q, name, qubits, _ in cir_list:
        depth_add = gate_depth[name] if name in gate_depth else 1
        depth_max = -1
        if num_q == 1: qubits = [qubits]
        for q in qubits:
            while q > len(qubit_to_depth) - 1: qubit_to_depth.append(0)
            if depth_max < qubit_to_depth[q]: depth_max = qubit_to_depth[q]
        for q in qubits:
            qubit_to_depth[q] = depth_max + depth_add
    depth = max(qubit_to_depth)
    # get success rate
    success_rate = None
    if error_matrix != []:
        success_rate = 1
        for num_q, name, q, _ in cir_list:
            if num_q > 2: raise()
            depth_add = gate_depth[name] if name in gate_depth else 1
            if num_q == 1:
                success_rate *= (1 - error_matrix[q][q]) ** depth_add
            if num_q == 2: 
                success_rate *= (1 - error_matrix[q[0]][q[1]]) ** depth_add
    if flag_print:
        print('The size is', size)
        print('The depth is', depth)
        if success_rate != None: print('The success rate is', success_rate)
    return size, depth, success_rate

'''
if __name__ == "__main__":
    file_path = 'D:/data/QASM example/test/201.qasm'
    dg = qasm_to_dg(file_path)
'''