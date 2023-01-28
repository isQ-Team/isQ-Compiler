# -*- coding: utf-8 -*-
"""
Created on Sun May 23 23:44:08 2021

@author: Xiangzhen Zhou
"""
import numpy as np

def qubit_convert(q_list):
    pass

class FrontCircuit():
    def __init__(self, DG, AG, front_cir_from=None):
        '''
        

        Parameters
        ----------
        map_list : TYPE
            index: logical qubits
            value: physical qubits
        DG : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        '''
        self.DG = DG
        self.AG = AG
        self.num_q_phy = len(AG)
        self.num_q_log = DG.num_q_log
        #self.unassigned_q = self.num_q_log
        if front_cir_from == None:
            self.num_remain_gates = len(DG)
            # initial mapping
            self.log_to_phy = [-1] * self.num_q_log
            self.phy_to_log = [-1] * self.num_q_phy
            # find first gates and front layer
            self.first_gates = [-1] * self.num_q_log
            self.front_layer = []
            current_nodes = []
            used_nodes = []
            for node in DG.nodes:
                if DG.in_degree[node] == 0:
                    current_nodes.append(node)
                    self.front_layer.append(node)
            i = 0
            while i < self.num_q_log and len(current_nodes) > 0:
                node = current_nodes.pop(0)
                used_nodes.append(node)
                q0, q1 = DG.nodes[node]['operand']
                if self.first_gates[q0] == -1:
                    self.first_gates[q0] = node
                    i += 1
                if self.first_gates[q1] == -1:
                    self.first_gates[q1] = node
                    i += 1
                for node_new in DG.successors(node):
                    if not node_new in used_nodes:
                        flag = True
                        for node_pre in DG.predecessors(node_new):
                            if not node_pre in used_nodes: flag = False
                        if flag == True: current_nodes.append(node_new)
            if i > self.num_q_log: raise()
        else:
            # copy
            self.num_remain_gates = front_cir_from.num_remain_gates
            # initial mapping
            self.log_to_phy = front_cir_from.log_to_phy.copy()
            self.phy_to_log = front_cir_from.phy_to_log.copy()
            # find first gates and front layer
            self.first_gates = front_cir_from.first_gates.copy()
            self.front_layer = front_cir_from.front_layer.copy()
            
    def __hash__(self):
        info = tuple(self.front_layer), tuple(self.log_to_phy)
        return hash(info)
    
    def assign_mapping_from_list(self, map_list):
        '''here the indices in map_list represent logical qubits'''
        for q_log in range(self.num_q_log):
            q_phy = map_list[q_log]
            self.log_to_phy[q_log] = q_phy
            self.phy_to_log[q_phy] = q_log
        exe_gates = self.execute_gates()
        return exe_gates
        
    def assian_mapping_naive(self):
        map_list = list(range(self.num_q_log))
        return self.assign_mapping_from_list(map_list)
        
    def swap(self, swap_phy):
        '''update via a SWAP'''
        q_phy0, q_phy1 = swap_phy
        q_log0, q_log1 = self.phy_to_log[q_phy0], self.phy_to_log[q_phy1]
        # update mapping
        self.phy_to_log[q_phy0] = q_log1
        self.phy_to_log[q_phy1] = q_log0
        if q_log0 != -1: self.log_to_phy[q_log0] = q_phy1
        if q_log1 != -1: self.log_to_phy[q_log1] = q_phy0
        # execute gates
        return self.execute_gates()
        
    def copy(self):
        return FrontCircuit(self.DG, self.AG, self)
    
    def swap_new(self, swap_phy):
        '''use a SWAP to get a new FrontCircuit object'''
        #new_cir = FrontCircuit(self.DG, self.AG, self)
        new_cir = self.copy()
        exe_gates = new_cir.swap(swap_phy)
        return new_cir, exe_gates
    
    def _executable(self, node):
        '''judge whether a node is executable'''
        q_log0, q_log1 = self.DG.nodes[node]['operand']
        q_phy0, q_phy1 = self.log_to_phy[q_log0], self.log_to_phy[q_log1]
        if q_phy0 != -1 and q_phy1 != -1:
            if (q_phy0, q_phy1) in self.AG.edges: return True
        return False
    
    def execute_front_layer(self):
        '''
        Execute all gates in the front layer regardless mapping
        However, we won't executable the following possible executable gates'
        '''
        layer = self.front_layer.copy()
        for node_dg in layer:
            self.execute_gate(node_dg)
        
    def execute_gates(self):
        '''find all executable gates and execute them'''
        exe_gates = []
        i = 0
        max_i = len(self.front_layer) - 1
        while i <= max_i:
            current_node = self.front_layer[i]
            # check cnot executable
            if self._executable(current_node):
                self.execute_gate_index(i)
                exe_gates.append(current_node)
                max_i = len(self.front_layer) - 1
            else:
                i += 1
        return exe_gates
    
    def execute_gate_index(self, front_layer_i):
        '''We only execute specified gate and will not execute its successors'''
        self.num_remain_gates -= 1
        exe_node = self.front_layer.pop(front_layer_i)
        q_exe0, q_exe1 = self.DG.nodes[exe_node]['operand']
        nodes_next = list(self.DG.successors(exe_node))
        self.first_gates[q_exe0], self.first_gates[q_exe1] = -1, -1
        # deal with the successors of executed node
        for node in nodes_next:
            for q in self.DG.nodes[node]['operand']:
                if self.first_gates[q] == -1: self.first_gates[q] = node
            q0, q1 = self.DG.nodes[node]['operand']
            if self.first_gates[q0] == self.first_gates[q1]:
                self.front_layer.append(node)
                
    def execute_gate(self, node_DG):
        front_layer_i = self.front_layer.index(node_DG)
        self.execute_gate_index(front_layer_i)
        
    def pertinent_swaps(self):
        '''
        Get pertinent swaps and their evaluations
        '''
        swaps_phy = []
        h_scores = []
        qubits_phy = []
        qubits_phy_other = [-1] * self.num_q_phy
        for node in self.front_layer:
            q0, q1 = self.DG.nodes[node]['operand']
            q0_phy = self.log_to_phy[q0]
            q1_phy = self.log_to_phy[q1]
            qubits_phy.extend([q0_phy, q1_phy])
            qubits_phy_other[q0_phy] = q1_phy
            qubits_phy_other[q1_phy] = q0_phy
                
        for swap in self.AG.edges:
            q0, q1 = swap
            flag = False
            current_score = 0
            # calculate score
            if q0 in qubits_phy:
                flag = True
                current_score += self.AG.shortest_length[q0][qubits_phy_other[q0]]\
                    - self.AG.shortest_length[q1][qubits_phy_other[q0]]
            if q1 in qubits_phy:
                flag = True
                current_score += self.AG.shortest_length[q1][qubits_phy_other[q1]]\
                    - self.AG.shortest_length[q0][qubits_phy_other[q1]]
            # add swap and score info
            if flag == True:
                swaps_phy.append(swap)
                h_scores.append(current_score)
        return swaps_phy, h_scores
    
    def print(self):
        gate_phy = []
        for node in self.front_layer:
            q0, q1 = self.DG.nodes[node]['operand']
            gate_phy.append((self.log_to_phy[q0], self.log_to_phy[q1]))
        print('mapping from log to phy:', self.log_to_phy)
        print('remaining gates:', self.num_remain_gates)
        print('front layer physical gates:', gate_phy)
        #print('qubits first gates:', self.first_gates)
        
    def print_front_layer(self):
        q = []
        for node in self.front_layer:
            q0, q1 = self.DG.nodes[node]['operand']
            q.append((self.log_to_phy[q0], self.log_to_phy[q1]))
        print(q)
        
    def get_future_cx_fix_num(self, num_cx):
        '''
        Get a specific number of unexecuted cx info
        (the corresponding operand physical qubits)
        '''
        first_gates_back_up = self.first_gates.copy()
        front_layer_back_up = self.front_layer.copy()
        num_remain_gates_back_up = self.num_remain_gates
        cx0 = []
        cx1 = []
        i = 0
        while i < num_cx and self.num_remain_gates > 0:
            i += 1
            node = self.front_layer[0]
            q0, q1 = self.DG.nodes[node]['operand']
            cx0.append(self.log_to_phy[q0])
            cx1.append(self.log_to_phy[q1])
            self.execute_gate(node)
        # restore information
        if len(cx0) > num_cx: raise()
        self.first_gates = first_gates_back_up
        self.front_layer = front_layer_back_up
        self.num_remain_gates = num_remain_gates_back_up
        return cx0, cx1

    def get_future_cx_fix_num2(self, num_cx):
        '''
        Get a specific number of unexecuted cx info
        (the corresponding operand physical qubits)
        this mehtod obtain gates layer by layer
        '''
        first_gates_back_up = self.first_gates.copy()
        front_layer_back_up = self.front_layer.copy()
        num_remain_gates_back_up = self.num_remain_gates
        cx0 = []
        cx1 = []
        i = 0
        while i < num_cx and self.num_remain_gates > 0:
            for node in self.front_layer.copy():
                i += 1
                q0, q1 = self.DG.nodes[node]['operand']
                cx0.append(self.log_to_phy[q0])
                cx1.append(self.log_to_phy[q1])
                self.execute_gate(node)
        # restore information
        self.first_gates = first_gates_back_up
        self.front_layer = front_layer_back_up
        self.num_remain_gates = num_remain_gates_back_up
        return cx0, cx1
    
    def get_future_cx_fix_num3(self, num_cx):
        '''
        Get a specific number of unexecuted cx info
        (the corresponding operand physical qubits)
        this mehtod obtain gates layer by layer and
        return tuples dividing gates according their layers
        '''
        first_gates_back_up = self.first_gates.copy()
        front_layer_back_up = self.front_layer.copy()
        num_remain_gates_back_up = self.num_remain_gates
        i = 0
        cx_total = []
        while i < num_cx and self.num_remain_gates > 0:
            cx_layer = []
            for node in self.front_layer.copy():
                i += 1
                q0, q1 = self.DG.nodes[node]['operand']
                cx0 = self.log_to_phy[q0]
                cx1 = self.log_to_phy[q1]
                cx_layer.append((cx0, cx1)) if cx0 <= cx1 else cx_layer.append((cx1, cx0))
                self.execute_gate(node)
            cx_layer.sort()
            cx_total.append(cx_layer)
        # restore information
        self.first_gates = first_gates_back_up
        self.front_layer = front_layer_back_up
        self.num_remain_gates = num_remain_gates_back_up
        return cx_total
    
    def check_equal(self, cir2):
        if ((self.log_to_phy == cir2.log_to_phy) and 
            (self.phy_to_log == cir2.phy_to_log) and
            (self.num_remain_gates == cir2.num_remain_gates) and
            (self.front_layer == cir2.front_layer) and
            (self.first_gates == cir2.first_gates)):
            return True
        else:
            return False
        
    def get_cir_matrix(self, num_layer):
        '''
        create a numpy matrix to represent the circuit with multi-layers containing
        only CNOT gates
        input:
            num_q_log -> total number of logical qubits. E.g., 4
            CNOT_list -> list of CNOT contains tuples showing input logical qubits
                         for corresponding CNOT gates.
                         E.g., [(0, 2), (3, 1), (2, 3) ...]
            output:
                [0 0 1 0
                 0 0 0 1
                 1 0 0 0
                 0 1 0 0]
                [0 0 0 0
                 0 0 0 0
                 0 0 0 1
                 0 0 1 0]
                ......
        '''
        cir_map = np.zeros([num_layer, self.num_q_phy, self.num_q_phy]).astype(np.float32)
        DG = self.DG
        cir = self.copy()
        i = 0
        while i < num_layer:
            for node_dg in cir.front_layer:
                q0, q1 = DG.nodes[node_dg]['operand']
                q0_phy, q1_phy = (cir.log_to_phy[q0], cir.log_to_phy[q1])
                cir_map[i][q0_phy][q1_phy], cir_map[i][q1_phy][q0_phy] = 1, 1
            # go to next layer
            if len(cir.front_layer) == 0: break
            i += 1
            cir.execute_front_layer()
        return cir_map, i