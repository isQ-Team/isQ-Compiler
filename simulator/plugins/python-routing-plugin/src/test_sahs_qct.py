# -*- coding: utf-8 -*-
"""
Created on Sun Aug  1 18:30:20 2021

@author: zhoux
"""

import numpy as np

def init_error_matrix(ag_matrix):
    '''exemplary error martix'''
    num_q = len(ag_matrix)
    m = np.zeros((num_q, num_q))
    i = 0
    flag2 = True
    flag = True
    for i in range(num_q):
        for j in range(i, num_q):
            if i == j:
                if flag2: m[i][j] = 0.0001
                else:
                    m[i][j] = 0.0005
                flag2 = not flag2
            else:
                if ag_matrix[i][j] == 0: continue
                if flag:
                    m[i][j], m[j][i] = 0.001, 0.001
                else:
                    m[i][j], m[j][i] = 0.001 * 50, 0.001 * 50
                i += 1
                if np.mod(i, 3) == 0: flag = not flag
    return m

def gen_grid_connectivity(l, w):
    '''Connectivity constraint for a grid QPU with length l and width w.'''
    num_q = l * w
    ag_matrix = np.zeros((num_q, num_q))
    edges = []
    length = l
    width = w
    for raw in range(width-1):
        for col in range(length-1):
            current_v = col + raw*length
            edges.append((current_v, current_v + 1))
            edges.append((current_v, current_v + length))
    for raw in range(width-1):
        current_v = (length - 1) + raw*length
        edges.append((current_v, current_v + length))
    for col in range(length-1):
        current_v = col + (width - 1)*length
        edges.append((current_v, current_v + 1))
    for (q0, q1) in edges:
        ag_matrix[q0][q1], ag_matrix[q1][q0] = 1, 1
    return ag_matrix


def QCT_SAHS(cir_in, ag_matrix, **args):
    '''
    https://ieeexplore.ieee.org/abstract/document/8970267
    The optional parameters (args) for SAHS QCT:
    display_state (default value False):
        if set to True, then we will display the progress during QCT
    method_ini_map (default value 'naive'):
        initial mapping for the input circuit, can be set to 'naive' or 
        'simulated_annealing'. 'simulated_annealing' is a heuristic algorithm
        to find a initial mapping and it will be slower than 'naive'.
        Currently, 'simulated_annealing' will cost much more time and is not
        stable.
    depth (default value 2):
        Search depth for QCT. If increased, better result will be found with an
        exponential time and space overhead
    prune_ratio (default value 0):
        Must be set between 0 and 1. A larger purning ratio will decrease the 
        running time and, at the mean time, degrade the quality of the result
    objective:
        'size' (default) or 'depth' or 'error'
        size: 
            minimize the # of added SWAP gates
        depth: 
            minimize the depth of the output circuit, assumining each gate 
            in the input circuit has depth 1 and SWAP in the output circuit 
            depth 3
        error:
            maximum the success rate of the output circuit. When this metric
            is chosen, error_matrix must be given
    error_matrix:
        a np square matrix in which each index [i][j] represent the error 
        probability for executing CX[i][j] on the target QPU.
    Return:
        circuit list, inital mapping, final mapping
    '''
    import time
    from cir_gen.interface import cir_list_to_dg, array_to_ag
    from sahs.sahs_search import SahsSearch
    # AG
    AG = array_to_ag(ag_matrix)
    # generate dependency graph
    DG = cir_list_to_dg(cir_in, len(AG))
    num_gates_in = len(cir_in)
    # init search tree
    search_tree = SahsSearch(AG, DG,
                             **args,)
    t_start = time.time()
    final_node = search_tree.qct()
    t_end = time.time()
    # print result
    res_gate_add = search_tree.nodes[final_node]['num_add_gates']
    res_gate_total = num_gates_in + res_gate_add
    res_time = t_end - t_start
    #print('Number of added SWAPs is', res_gate_add/3)
    #print('Number of added gates is', res_gate_add)
    #print('Number of all gates is', res_gate_total)
    #print('Total time is', res_time)
    return search_tree.to_cir_list(), search_tree.init_map, search_tree.get_mappings(final_node)[0]

def get_grid_from_edges(num_q, edges):
    ag_matrix = np.zeros((num_q, num_q))
    for (q0, q1) in edges:
        if q0 > num_q or q1 > num_q:
            return []
        ag_matrix[q0-1][q1-1], ag_matrix[q1-1][q0-1] = 1, 1
    return ag_matrix

def get_cir_in_from_qcis(qcis_str):
    cir_in = []
    qcis = qcis_str.split('\n')
    for command in qcis:
        #print(command)
        command = command.strip().split(' ')
        if len(command) == 0:
            continue
        gate = command[0]
        if gate == 'M':
            continue
        q = int(command[1][1:])-1
        qn = 1
        if len(command) > 2:
            q = (q, int(command[2][1:])-1)
            qn = 2
        cir_in.append((qn, gate, q, []))
    
    return cir_in

def convert_swap(q):
    (q2, q1) = q
    swap = []
    for i in range(3):
        q1, q2 = q2, q1
        swap.append("H Q{}".format(q2+1))
        swap.append("CZ Q{} Q{}".format(q1+1, q2+1))
        swap.append("H Q{}".format(q2+1))
    return "\n".join(swap)

def get_qcis_from_cir_out(cir_out):
    
    qcis = []

    for (qn, gate, q, _) in cir_out:
        if gate == "SWAP":
            qcis.append(convert_swap(q))
        else:
            if qn == 1:
                qcis.append("{} Q{}".format(gate, q+1))
            else:
                qcis.append("{} Q{} Q{}".format(gate, q[0]+1, q[1]+1))
        
    return "\n".join(qcis)

if __name__ == '__main__':
    import os
    import sys
    from json import loads
    # generate example connectivity
    data = loads(sys.stdin.read())
    if 'QCIS_DONT_ROUTE' in os.environ and os.environ['QCIS_DONT_ROUTE']!='0':
        print(data["qcis"])
        sys.exit(0)
    ag_matrix = get_grid_from_edges(data["qbit_num"], data["topo"])
    #ag_matrix = gen_grid_connectivity(4, 4)
    #qasm_path = './qasm_circuits/'
    
    # set objective
    objective = 'size' # 'size' or 'depth' or 'error'
    # init depth of each gate
    gate_depth = {'SWAP': 3}
    display_state = 0
    # init example error_matrix
    error_matrix = init_error_matrix(ag_matrix)

    cir_in = get_cir_in_from_qcis(data["qcis"])

    init_map = 'naive'
    if "init_map" in data:
        init_map = data['init_map']
    #print(cir_in)

    cir_out, init_map_list, final_map_list = QCT_SAHS(cir_in, ag_matrix,
                                                          objective=objective,
                                                          error_matrix=error_matrix,
                                                          display_state=display_state,
                                                          init_map = init_map)
    
    qcis_out = get_qcis_from_cir_out(cir_out)
    sys.stdout.write(qcis_out)

    '''
    # get circuits from qasm_path
    QASM_files = os.listdir(qasm_path)
    #QASM_files = ["qft_5.qasm"]
    print('# circuits is', len(QASM_files))
    print(QASM_files)
    
    total_size = 0
    total_deptn = 0
    total_succ = 0
    for QASM in QASM_files:
        # QCT circuits one-by-one
        print('circuit name is', QASM)
        QASM_file = qasm_path + QASM
        ## generate circuit list from QASM file
        cir_in, _ = qasm_to_cir_list(QASM_file)
        print(cir_in)
        ## QCT
        cir_out, init_map_list, final_map_list = QCT_SAHS(cir_in, ag_matrix,
                                                          objective=objective,
                                                          error_matrix=error_matrix,
                                                          display_state=display_state,)
        ## print circut attributes
        size, depth, succ_rate = cir_list_attributes(cir_out, 
                                                     gate_depth=gate_depth, 
                                                     flag_print=True,
                                                     error_matrix=error_matrix)
        total_size += size
        total_deptn += depth
        total_succ += succ_rate
        ## EC
        if ag_matrix.shape[0] < 10:
            # we only do EC for circuits with less than 10 qubits
            eq = equivalence_checker_qiskit(cir_in,
                                            cir_out, final_map_list,
                                            ag_matrix)
            print('Result of equivalence checking:', eq)
        print('----------------------')
        print()
    print('Total size count is', total_size)
    print('Total depth count is', total_deptn)
    print('Total success rate is', total_succ)
    '''