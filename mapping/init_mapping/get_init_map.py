# -*- coding: utf-8 -*-
"""
Created on Fri Jul 23 16:04:46 2021

@author: zhoux
"""

legal_methods = {'naive', 'simulated_annealing'}

def get_init_map(DG, AG, map_method='naive'):
    '''
    Return
        map_list: represents a mapping in which indices and values stand for 
                  logical and physical qubits.
    '''
    if not map_method in legal_methods:
        raise(Exception("Unsupported method {} for initial mapping".format(map_method)))
    if map_method == 'naive':
        return list(range(len(AG)))
    if map_method == 'simulated_annealing':
        from .sa_mapping import InitialMapSimulatedAnnealingWeighted
        return InitialMapSimulatedAnnealingWeighted(DG, AG)
    