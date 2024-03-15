# -*- coding: utf-8 -*-

"""
Indicated implementation of computation of k shortest paths between two nodes
in networkx library.

Ref: https://networkx.github.io/documentation/networkx-1.10/reference/generated
     /networkx.algorithms.simple_paths.shortest_simple_paths.html
"""

import networkx as netx

from itertools import islice

def k_shortest_paths(G, source, target, k, weight=None):
    return list(islice(netx.shortest_simple_paths(
            G, source, target, weight=weight), k))

 