# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 15:03:29 2021

@author: zhoux
"""
import numpy as np
from front_circuit import FrontCircuit
from networkx import DiGraph

def CalCostMatrixWeighted(cost_m, current_sol, shortest_length_G, num_q):
    cost_total = 0
    for q1_log in range(num_q):
        for q2_log in range(num_q):
            q1_phy, q2_phy = current_sol[q1_log], current_sol[q2_log]
            num_swap = shortest_length_G[q1_phy][q2_phy] - 1
            cost_total += num_swap * cost_m[q1_log][q2_log]
    return cost_total

def InitialCostMatrixWeighted(DG, AG, method=2, add_weight=False):
    num_q = len(AG)
    cost_m = np.zeros((num_q, num_q))
    cir = FrontCircuit(DG, AG)
    if method == 1:
        '''method 1, exponential decay'''
        weight = 1
        decay = 0.99 # default 0.99
        while len(cir.front_layer) != 0:
            current_nodes = cir.front_layer
            for node in current_nodes:
                op = DG.nodes[node]['operation']
                qubits = op.involve_qubits
                cost_m[qubits[0][1]][qubits[1][1]] += weight
                #cost_m[qubits[1][1]][qubits[0][1]] += weight
            cir.execute_front_layer()
            weight = weight * decay
    if method == 2:
        '''method 2, linear decay'''
        num_CX = len(DG.nodes())
        num_CX_current = num_CX
        weight = 1
        while len(cir.front_layer) != 0:
            weight = num_CX_current / num_CX
            current_nodes = cir.front_layer
            num_CX_current -= len(current_nodes)
            for node in current_nodes:
                op = DG.nodes[node]['operand']
                if add_weight == True:
                    flag = 1
                    '''if comment the following, we ignore the successive CX'''
                    if DG.out_degree(node) == 1:
                        qubits = op
                        op_next = DG.nodes[list(DG.successors(node))[0]]['operand']
                        qubits_next = op_next
                        if qubits[0] == qubits_next[0] and qubits[1] == qubits_next[1]:
                            flag = 0
                        if qubits[0] == qubits_next[1] and qubits[1] == qubits_next[0]:
                            flag = 0 
                    
                    DG.nodes[node]['weight'] = weight * flag
                qubits = op
                if add_weight == True:
                    cost_m[qubits[0]][qubits[1]] += DG.nodes[node]['weight']
                else:
                    cost_m[qubits[0]][qubits[1]] += weight
                #cost_m[qubits[1][1]][qubits[0][1]] += weight
            cir.execute_front_layer()
    if method == 3:
        '''method 3, weighted cos decay'''
        num_CX = len(DG.nodes())
        num_CX_current = 0
        weight = 1
        while len(cir.front_layer) != 0:
            weightt = num_CX_current / num_CX
            weightt = np.power(weightt, 1)
            weight = (np.cos(np.pi * weightt)+1) / 2
            current_nodes = cir.front_layer
            num_CX_current += len(current_nodes)
            for node in current_nodes:
                op = DG.nodes[node]['operation']
                qubits = op.involve_qubits
                cost_m[qubits[0][1]][qubits[1][1]] += weight
                #cost_m[qubits[1][1]][qubits[0][1]] += weight
            cir.execute_front_layer()
    if method == 4:
        '''method 4, no decay'''
        num_CX = len(DG.nodes())
        num_CX_current = num_CX
        weight = 1
        while len(cir.front_layer) != 0:
            current_nodes = cir.front_layer
            num_CX_current -= len(current_nodes)
            for node in current_nodes:
                op = DG.nodes[node]['operation']
                if add_weight == True:
                    #print(DG.out_degree(node))
                    flag = 1
                    '''if comment the following, we ignore the successive CX'''
                    if DG.out_degree(node) == 1:
                        qubits = op.ToTuple()
                        op_next = DG.nodes[list(DG.successors(node))[0]]['operation']
                        qubits_next = op_next.ToTuple()
                        if qubits[0] == qubits_next[0] and qubits[1] == qubits_next[1]:
                            flag = 0
                        if qubits[0] == qubits_next[1] and qubits[1] == qubits_next[0]:
                            flag = 0                        
                    
                    op.weight = weight * flag
                qubits = op.involve_qubits
                if isinstance(qubits[0], int):
                    qubits[0] = qubits[0], qubits[0]
                    qubits[1] = qubits[1], qubits[1]
                if add_weight == True:
                    cost_m[qubits[0][1]][qubits[1][1]] += op.weight
                else:
                    cost_m[qubits[0].index][qubits[1].index] += weight
                #cost_m[qubits[1][1]][qubits[0][1]] += weight
            cir.execute_front_layer()
    return cost_m

def initpara():
    '''Initialize parameters for simulated annealing'''
    alpha = 0.98
    t = (1,100)#(1,100)
    markovlen = 100
    return alpha,t,markovlen

def InitialMapSimulatedAnnealingWeighted(DG,
                                         AG,
                                         start_map=None,
                                         convergence=False,):
    '''
    This function is modified from "https://blog.csdn.net/qq_34798326/article/details/79013338"
    
    Return
        solutionbest: represents a mapping in which indices and values stand
                      for logical and physical qubits.
    '''
    shortest_length_G = AG.shortest_length
    num_q = len(AG.nodes()) # num of physical qubits
    if convergence == True:
        temp = []
        solution = []
        solution_best = []
    if start_map == None: start_map = list(range(num_q))
    if len(start_map) != len(AG.nodes()):
        '''
        if logical qubits is less than physical, we extend logical qubit to
        ensure the completeness and delete added qubits at the end of the
        algorithm
        '''
        for v in AG.nodes():
            if not v in start_map: start_map.append(v)
    '''gen cost matrix'''
    cost_m = InitialCostMatrixWeighted(DG, AG)
    '''Simulated Annealing'''
    solutionnew = start_map
    num = len(start_map)
    #valuenew = np.max(num)
    solutioncurrent = solutionnew.copy()
    valuecurrent = 99000  #np.max这样的源代码可能同样是因为版本问题被当做函数不能正确使用，应取一个较大值作为初始值
    
    #print(valuecurrent)
    
    solutionbest = solutionnew.copy()
    valuebest = 99000 #np.max
    alpha,t2,markovlen = initpara()
    t = t2[1]#temperature
    result = [] #记录迭代过程中的最优解
    
    while t > t2[0]:
        for i in np.arange(markovlen):
            #下面的两交换和三角换是两种扰动方式，用于产生新解
            if np.random.rand() > 0.5:# 交换路径中的这2个节点的顺序
                # np.random.rand()产生[0, 1)区间的均匀随机数
                while True:#产生两个不同的随机数
                    loc1 = np.int(np.around(np.random.rand()*(num-1)))
                    loc2 = np.int(np.around(np.random.rand()*(num-1)))
                    ## print(loc1,loc2)
                    if loc1 != loc2:
                        break
                solutionnew[loc1],solutionnew[loc2] = solutionnew[loc2],solutionnew[loc1]
            else: #三交换
                while True:
                    loc1 = np.int(np.around(np.random.rand()*(num-1)))
                    loc2 = np.int(np.around(np.random.rand()*(num-1))) 
                    loc3 = np.int(np.around(np.random.rand()*(num-1)))
                    if((loc1 != loc2)&(loc2 != loc3)&(loc1 != loc3)):
                        break
                # 下面的三个判断语句使得loc1<loc2<loc3
                if loc1 > loc2:
                    loc1,loc2 = loc2,loc1
                if loc2 > loc3:
                    loc2,loc3 = loc3,loc2
                if loc1 > loc2:
                    loc1,loc2 = loc2,loc1
                #下面的三行代码将[loc1,loc2)区间的数据插入到loc3之后
                tmplist = solutionnew[loc1:loc2].copy()
                solutionnew[loc1:loc3-loc2+1+loc1] = solutionnew[loc2:loc3+1].copy()
                solutionnew[loc3-loc2+1+loc1:loc3+1] = tmplist.copy()  
            valuenew = CalCostMatrixWeighted(cost_m, solutionnew,
                                             shortest_length_G, num_q)
           # print (valuenew)
            if valuenew < valuecurrent: #接受该解
                #更新solutioncurrent 和solutionbest
                valuecurrent = valuenew
                solutioncurrent = solutionnew.copy()
                #renew best solution
                if valuenew < valuebest:
                    valuebest = valuenew
                    solutionbest = solutionnew.copy()
            else:#按一定的概率接受该解
                if np.random.rand() < np.exp(-(valuenew-valuecurrent)/t):
                    valuecurrent = valuenew
                    solutioncurrent = solutionnew.copy()
                else:
                    solutionnew = solutioncurrent.copy()

            if convergence == True:
                temp.append(t)
                solution.append(valuecurrent)
                solution_best.append(valuebest)
        
        t = alpha*t
        #print(valuebest)
        result.append(valuebest)
    '''draw convergence graph'''
    #if convergence == True:
    #    figure_fig = plt.figure()
    #    plt.grid()
    #    plt.xlabel('Times of Iteration')
    #    plt.ylabel('Cost of States')
    #    plt.plot(solution)
    #    plt.plot(solution_best)
    #    figure_fig.savefig('simulated annealing convergence.eps', format='eps', dpi=1000)
        
    return solutionbest