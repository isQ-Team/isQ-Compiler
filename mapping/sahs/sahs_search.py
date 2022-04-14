# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 12:50:41 2021

@author: zhoux
"""
import networkx as nx
from networkx import DiGraph
import numpy as np
import time
from front_circuit import FrontCircuit
from init_mapping.get_init_map import get_init_map

'''parameters for the search process'''
display_state_default = 0 # whether we print the internal states during search process
delete_trivival_swap = 1 # should we ignore duplicated swaps to avoid dead loop?
weights_default = [1, 1, 0.8, 0.6, 0.4] # [1, 1, 0.8, 0.6, 0.4]
depth_default = 2
prune_ratio_default = 0
objective_default = 'size'
    
MCTree_key_words = ['weights', 'depth', 'display_state', 'init_map',
                    'num_layer_ann', 'prune_ratio', 'objective',
                    'error_matrix']

class SahsSearch(DiGraph):
    def __init__(self, AG, DG, **args):
        '''
        swap_combination is a list of swaps to be considered
        T: ratio for node evaluation
        node_count: index for newly added node
        '''
        super().__init__()
        # check key words
        for i in args:
            if not i in MCTree_key_words:
                raise(Exception('Unsupported keyword {}'.format(i)))
        # set parameters
        #self.use_remote = use_remote
        
        self.node_count = 0
        self.AG = AG
        self.num_q_phy = len(AG)
        self.num_q_log = DG.num_q_log
        self.DG = DG
        self.max_length = nx.diameter(AG)
        self.fallback_value = self.num_q_phy * 2 #self.max_length * 10
        self.fallback_count = 0
        self.finish_nodes = []
        if 'weights' in args:
            self.weights = args['weights']
        else:
            self.weights = weights_default
        if 'depth' in args:
            self.depth = args['depth']
        else:
            self.depth = depth_default
        if 'display_state' in args:
            self.display_state = args['display_state']
        else:
            self.display_state = display_state_default
        if 'init_map' in args:
            self.init_map = get_init_map(DG, AG, args['init_map'])
        else:
            # we use naive mapping as default
            self.init_map = list(range(self.num_q_phy))
        if 'prune_ratio' in args:
            self.prune_ratio = args['prune_ratio']
        else:
            self.prune_ratio = prune_ratio_default
        if 'objective' in args:
            self.objective = args['objective']
        else:
            self.objective = objective_default
        if self.objective == 'size':
            self.get_total_cost = self.get_total_cost_size
        if self.objective == 'depth':
            self.get_total_cost = self.get_total_cost_depth
        if self.objective == 'error':
            self.get_total_cost = self.get_total_cost_error
        if self.objective == 'error':
            error_matrix = args['error_matrix']
            #self.weight_matrix = np.log(error_matrix + 0.0000000000001)
            self.weight_matrix = -1 * np.log(1 - error_matrix)
            # calculate shortest distance via weight matrix
            for edge in self.AG.edges:
                self.AG.edges[edge]['weight'] = self.weight_matrix[edge[0]][edge[1]]
                self.shortest_length_AG = dict(nx.shortest_path_length(self.AG, 
                                                                       source=None,
                                                                       target=None,
                                                                       weight='weight',
                                                                       method='dijkstra'))
                self.shortest_path_AG = nx.shortest_path(self.AG, 
                                                         source=None, target=None, 
                                                         weight='weight', method='dijkstra')
        else:
            self.shortest_length_AG = AG.shortest_length
            self.shortest_path_AG = AG.shortest_path
            
        # initialize the first node
        self.root_node = self.add_node(father_node=None)
        self.init_node = self.root_node
        self.leaves = [self.root_node]
    
    def add_node(self, father_node, added_swap=None,
                added_remote=None):
        '''
        added_swap: list like [(in1, in2),...]
        num_remain_gates: number of unexecuted CNOT gates in logical circuit
        return: generated node number
        '''
        new_node = self.node_count
        self.node_count += 1
        if father_node == None:
            # root node
            cir = FrontCircuit(self.DG, self.AG)
            exe_gates = cir.assign_mapping_from_list(self.init_map)
            self.add_nodes_from([new_node])
            self.nodes[new_node]['num_add_gates'] = 0
            self.nodes[new_node]['father_node'] = None
        else:
            cir = self.nodes[father_node]['circuit'].copy()
            exe_gates = cir.swap(added_swap)
            self.add_nodes_from([new_node])
            self.add_edge(father_node, new_node)
            self.nodes[new_node]['num_add_gates'] = self.nodes[father_node]['num_add_gates'] + 3
        self.nodes[new_node]['circuit'] = cir
        self.nodes[new_node]['father_node'] = father_node
        '''initialize scores for size optimisation'''
        self.nodes[new_node]['added_swap'] = added_swap
        self.nodes[new_node]['execute_gates'] = exe_gates
        self.nodes[new_node]['local_score'] = len(exe_gates)
        self.nodes[new_node]['num_remain_gates'] = cir.num_remain_gates
        self.nodes[new_node]['root'] = None
        self.nodes[new_node]['cost_h'] = self.get_h_cost(new_node)
        self.nodes[new_node]['cost_total'] = self.get_total_cost(new_node)
        # check finish
        if self.nodes[new_node]['num_remain_gates'] == 0:
            self.finish_nodes.append(new_node)
        return new_node
    
    def get_father(self, node):
        return self.nodes[node]['father_node']
    
    def get_execute_nodes_dg(self, node):
        return self.nodes[node]['execute_gates']
    
    def get_add_swap(self, node):
        return self.nodes[node]['added_swap']

    def get_mappings(self, node):
        cir = self.nodes[node]['circuit']
        return cir.log_to_phy, cir.phy_to_log
    
    def get_total_cost_size(self, node):
        return self.nodes[node]['cost_h'] + self.nodes[node]['num_add_gates']
    
    def get_total_cost_depth(self, node):
        father = self.get_father(node)
        if father == None:
            qubit_depth = [0] * self.num_q_phy
            depth_add_fatehr = 0
        else:
            qubit_depth = self.nodes[father]['qubit_depth'].copy()
            depth_add_fatehr = self.nodes[father]['depth_add']
        # update depth information
        ## consider swap
        depth_before = max(qubit_depth)
        if father != None:
            q0, q1 = self.get_add_swap(node)
            depth_max = max(qubit_depth[q0], qubit_depth[q1])
            qubit_depth[q0] = depth_max + 3
            qubit_depth[q1] = depth_max + 3
        depth_add = max(qubit_depth) - depth_before + depth_add_fatehr
        self.nodes[node]['depth_add'] = depth_add
        ## consider logcal
        cir = self.nodes[node]['circuit']
        for node_dg in self.get_execute_nodes_dg(node):
            q0, q1 = self.DG.nodes[node_dg]['operand']
            q0, q1 = cir.log_to_phy[q0], cir.log_to_phy[q1]
            depth_max = max(qubit_depth[q0], qubit_depth[q1])
            qubit_depth[q0] = depth_max + 1 + \
                len(self.DG.nodes[node_dg]['single_gates0'])
            qubit_depth[q1] = depth_max + 1 + \
                len(self.DG.nodes[node_dg]['single_gates1'])
        # update node attribute
        self.nodes[node]['qubit_depth'] = qubit_depth
        depth = max(qubit_depth)
        # calculate cost_total
        add_cx = self.nodes[node]['num_add_gates']
        return self.nodes[node]['cost_h'] + depth_add / 1
    
    def get_total_cost_error(self, node):
        father = self.get_father(node)
        if father == None:
            cost_g = 0
        else:
            cost_g = self.nodes[father]['cost_g']
        # update depth information
        ## consider swap
        if father != None:
            q0, q1 = self.get_add_swap(node)
            cost_g += (self.weight_matrix[q0][q1] * 3)
        ## consider logcal gates
        cir = self.nodes[node]['circuit']
        for node_dg in self.get_execute_nodes_dg(node):
            # CX
            q0, q1 = self.DG.nodes[node_dg]['operand']
            q0, q1 = cir.log_to_phy[q0], cir.log_to_phy[q1]
            cost_g += self.weight_matrix[q0][q1]
            # single-qubit gate
            cost_g += (self.weight_matrix[q0][q0] * \
                len(self.DG.nodes[node_dg]['single_gates0']))
            cost_g += (self.weight_matrix[q1][q1] * \
                len(self.DG.nodes[node_dg]['single_gates1']))
        # update node attribute
        self.nodes[node]['cost_g'] = cost_g
        # calculate cost_total
        return self.nodes[node]['cost_h'] + cost_g * 1
    
    def get_h_cost(self, node):
        cir = self.nodes[node]['circuit'].copy()
        cost_h = 0
        # get h cost for the last layer
        cost_h += cir.num_remain_gates * self.weights[-1] * (self.num_q_phy - 1) * 3
        for i in range(len(self.weights)):
            w = self.weights[i]
            # get cost layer by layer
            for node_dg in cir.front_layer:
                # get cost node by node
                q_log0, q_log1 = self.DG.nodes[node_dg]['operand']
                q_phy0, q_phy1 = cir.log_to_phy[q_log0], cir.log_to_phy[q_log1]
                cost_h += (self.shortest_length_AG[q_phy0][q_phy1] - 1) * w * 3
            # go to next layer
            layer = cir.front_layer.copy()
            for node_dg in layer:
                cir.execute_gate(node_dg)
        return cost_h
    
    def get_h_cost2(self, node):
        cir_list = self.get_future_cir_list(node)
        cost_h = 0
        for i in range(len(self.weights)):
            w = self.weights[i]
            # get cost layer by layer
            for q0, q1 in cir_list[i]:
                # get cost node by node
                cx_cost = (self.shortest_length_AG[q0][q1] - 1) * 3
                if cx_cost == 0: raise()
                cost_h += cx_cost * w 
        return cost_h   
    
    def get_future_cir_list(self, node):
        if 'cir_list' in self.nodes[node] and self.nodes[node]['root'] == self.root_node:
            return self.nodes[node]['cir_list']
        if node == self.root_node:
            cir = self.nodes[node]['circuit'].copy()
            cir_list = [[]] * len(self.weights)
            for i in range(len(self.weights)):
                # get cx layer by layer
                for node_dg in cir.front_layer:
                    # get cost node by node
                    q_log0, q_log1 = self.DG.nodes[node_dg]['operand']
                    q_phy0, q_phy1 = cir.log_to_phy[q_log0], cir.log_to_phy[q_log1]
                    cir_list[i].append((q_phy0, q_phy1))
                # go to next layer
                layer = cir.front_layer.copy()
                for node_dg in layer:
                    cir.execute_gate(node_dg)
            self.nodes[node]['cir_list'] = cir_list
            self.nodes[node]['root'] = self.root_node
            return cir_list
        father = self.get_father(node)
        cir_list_father = self.get_future_cir_list(father)
        cir_list = [[]] * len(self.weights)
        # execute swap
        swap = self.get_add_swap(node)
        for layer_i in range(len(self.weights)):
            for q0, q1 in cir_list_father[layer_i]:
                if q0 == swap[0]: q0 = swap[1]
                else:
                    if q0 == swap[1]: q0 = swap[0]
                if q1 == swap[0]: q1 = swap[1]
                else: 
                    if q1 == swap[1]: q1 = swap[0]
                cx = (q0, q1)
                if not cx in self.AG.edges: cir_list[layer_i].append(cx)
        self.nodes[node]['cir_list'] = cir_list
        self.nodes[node]['root'] = self.root_node
        return cir_list
    
    def expand_node_via_swap(self, node, swap):
        added_node = self.add_node(node, swap)
        return added_node
    
    def expansion(self, node):
        '''
        expand a node via all non-trivival swaps and then do backpropogation
        '''
        if self.out_degree[node] != 0: raise(Exception('Expanded node already has son nodes.'))
        
        if self.nodes[node]['num_remain_gates'] == 0:
            # we can't expand leaf
            return []
        
        swaps = self.nodes[node]['circuit'].pertinent_swaps()
        swaps = swaps[0]
        
        if delete_trivival_swap == 1:
            '''
            check wehther expanded node does not exe any CX gates, if yes, we delete
            the swap that the father of expanded node does to avoid dead loop
            '''
            if self.nodes[node]['local_score'] == 0:
                swap_delete = self.nodes[node]['added_swap']
                if swap_delete in swaps: swaps.remove(swap_delete)
        
        added_nodes = []
        for swap in swaps:
            add_node = self.expand_node_via_swap(node, swap)
            if add_node != None: added_nodes.append(add_node)
            
        if self.prune_ratio > 0:
            if len(self.finish_nodes) > 0:
                return added_nodes
            added_costs = []
            for add_node in added_nodes:
                added_costs.append(self.nodes[add_node]['cost_total'])
            nodes_delete = []
            num_prune = self.prune_ratio * len(swaps)
            while len(nodes_delete) < num_prune:
                i = np.argmax(added_costs)
                added_costs[i] = 0
                nodes_delete.append(added_nodes[i])
            for del_node in nodes_delete:
                self.remove_node(del_node)
                added_nodes.remove(del_node)
        return added_nodes
    
    def expand_leaves(self):
        '''
        flag_ann: if True, then add 
        '''
        leaves_next = []
        for node in self.leaves:
            if node in self.nodes:
                leaves_next.extend(self.expansion(node))
        self.leaves = leaves_next
                
    def pick_best_son(self):
        '''
        Find the best son node according to leaf nodes
        '''
        min_cost = 0
        best_leaf = None
        for node in self.leaves:
            if not node in self.nodes: continue
            current_cost = self.nodes[node]['cost_total']
            if (best_leaf == None or current_cost < min_cost):
                best_leaf = node
                min_cost = current_cost
        best_son = best_leaf
        father = self.get_father(best_son)
        while father != self.root_node:
            best_son = father
            father = self.get_father(best_son)
        return best_son

    def delete_nodes(self, nodes):
        '''delete nodes and all its successors'''
        for node in nodes:
            '''delete'''
            T_succ = nx.dfs_tree(self, node)
            self.remove_nodes_from(T_succ.nodes)
    
    def fallback(self):
        if self.display_state == 1: print('Fallback!')
        deleted_nodes = []
        start_node = self.root_node
        '''find the initial node for fallback'''
        while self.nodes[start_node]['local_score'] == 0:
            father = self.get_father(start_node)
            if father == None: break
            deleted_nodes.append(start_node)
            start_node = father
        '''extract swaps list'''
        executable_vertex = self.nodes[start_node]['circuit'].front_layer
        log_to_phy = self.nodes[start_node]['circuit'].log_to_phy
        min_CX_dis = 1000
        for v in executable_vertex:
            CX = self.DG.nodes[v]['operand']
            CX_phy = log_to_phy[CX[0]], log_to_phy[CX[1]]
            currant_CX_dis = self.shortest_length_AG[CX_phy[0]][CX_phy[1]]
            if currant_CX_dis < min_CX_dis:
                min_CX_dis = currant_CX_dis
                chosen_CX_phy = CX_phy
        path = self.shortest_path_AG[chosen_CX_phy[0]][chosen_CX_phy[1]].copy()
        min_CX_dis = len(self.shortest_path_AG[chosen_CX_phy[0]][chosen_CX_phy[1]]) - 1
        #num_swap = int(np.ceil(min_CX_dis/2))
        num_swap = int(min_CX_dis - 1)
        '''set new root node and delete redunant nodes'''
        self.root_node = start_node
        self.delete_nodes(deleted_nodes)
        '''add swaps'''
        flag = True
        for i in range(num_swap):
            if flag == True:
                added_swap = path.pop(0), path[0]
            else:
                added_swap = path.pop(), path[-1]
            flag = not flag
            added_node = self.expand_node_via_swap(self.root_node, added_swap)
            self.root_node = added_node
        # update nodes info
        self.leaves = [self.root_node]
        for _ in range(self.depth-1):
            self.expand_leaves()
        if self.nodes[self.root_node]['local_score'] == 0:
            '''
            if the newly added node still can't execute any CX, there is sth
            wrong with the fallback procedure
            '''
            raise(Exception('Fallback error!'))
    
    def decision(self, flag_delete=True):
        '''
        choose one leaf node, delete all other leaf nodes of its father node
        '''
        best_son = self.pick_best_son()
        if flag_delete:
            # delete residual nodes
            deleted_nodes = list(self.successors(self.root_node))
            deleted_nodes.remove(best_son)
            self.delete_nodes(deleted_nodes)
        '''update root node'''
        #print('pre root is', self.root_node)
        self.root_node = best_son
        #print('Chosen next node is %d' %best_son)
        if self.display_state == True:
            print('\r%d gates unfinished'
                  %self.nodes[self.root_node]['num_remain_gates'],end='')
            #print('added swap is %s\n' %str(self.nodes[best_son]['added_swap']))
        '''update fallback count'''
        if self.nodes[best_son]['local_score'] == 0:
            self.fallback_count += 1
        else:
            self.fallback_count = 0
        if self.fallback_count >= self.fallback_value:
            #raise()
            self.fallback()
            self.fallback_count = 0
            return None
            
    def qct(self):
        for _ in range(self.depth):
            self.expand_leaves()
        while len(self.finish_nodes) == 0:
            self.decision()
            self.expand_leaves() 
        #nx.draw(self)
        #self.print_swaps()
        while self.nodes[self.root_node]['num_remain_gates'] > 0:
            self.decision()
        return self.root_node
  
    def print_swaps(self):
        node = self.finish_nodes[0]
        swaps = []
        while node != self.init_node:
            swaps.append(self.nodes[node]['added_swap'])
            node = self.get_father(node)
        swaps.reverse()
        print(swaps)

     
    def print_node_args(self, node, names):
        print('  node %d' %node)
        for name in names:
            print('    %s is %s' %(name, self.nodes[node][name]))
    
    def print_son_args(self, father_node, names_son, names_father=[]):
        if not isinstance(names_son, list) and not isinstance(names_son, tuple):
            raise(Exception('names argument must be list or tuple, but it is %s'
                            %type(names_son)))
        if not isinstance(names_father, list) and not isinstance(names_father, tuple):
            raise(Exception('names argument must be list or tuple, but it is %s'
                            %type(names_father)))            
        print('father node is %d' %father_node)
        sons = self.nodes[father_node]['son_nodes']
        for name in names_father:
            print('    %s is %s' %(name, self.nodes[father_node][name]))
        print('all son nodes of %d' %father_node)
        for son in sons:
            self.PrintNodeArgs(son, names_son)
            
# What follow are circuit representation converters
    def to_cir_list(self):
        cir_list = []
        node = self.init_node
        # add single_gates_before_first_layer
        for gate in self.DG.single_gates_before_first_layer:
            q = self.nodes[node]['circuit'].log_to_phy[gate[2]]
            cir_list.append((1, gate[1], q, gate[3]))
        while True:
            cir = self.nodes[node]['circuit']
            added_swap = self.nodes[node]['added_swap']
            exe_nodes_dg = self.nodes[node]['execute_gates']
            # add swap
            if added_swap != None:
                cir_list.append((2, 'SWAP', (added_swap[0], added_swap[1]), []))
            # add two- and single-qubit gates
            for node_dg in exe_nodes_dg:
                ## add two-qubit gate
                two_gate = self.DG.nodes[node_dg]['gate']
                q0_log, q1_log = self.DG.nodes[node_dg]['operand']
                q0_phy, q1_phy = cir.log_to_phy[q0_log], cir.log_to_phy[q1_log]
                cir_list.append((2, two_gate[1], (q0_phy, q1_phy), []))
                ## add single-qubit gates
                for single_gate in self.DG.nodes[node_dg]['single_gates0']:
                    cir_list.append((1, single_gate[1], q0_phy, single_gate[3]))
                for single_gate in self.DG.nodes[node_dg]['single_gates1']:
                    cir_list.append((1, single_gate[1], q1_phy, single_gate[3]))
            # go to the next node
            if self.out_degree(node) == 0: break
            if self.out_degree(node) > 2:
                raise()
            node = list(self.successors(node))[0]
        return cir_list