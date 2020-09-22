import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
plt.style.use('seaborn')

def number_generator(constants, distribution):
    """
    Generates a number according to a specific distribution and its specified constants.
    Options for distributions are:
    - Newmans Powerlaw
    - Geometric
    - Exponential

    Newmans Powerlaw:
    Uses the same algorithm Newman in his paper did (page 9) to return a 
    random number of the powerlaw distribution with a cutoff and an 
    exponent theta.
    Takes as arguments the cutoff value and the exponent.

    Geometric:
    Generates a k in the degree distribution: p_k = (1-a)a^k. 
    Found on Newman book page 580.
    The method for generating this number was taken 
    from the Newman article, page 9. 
    Takes as argument the constant a. 

    Exponential:
    Generates a k in the degree distribution:
    p_k = (1 - e**(-1/kappa)) e**(-k/kappa).
    Found in Nemwman particle page 4.
    Takes as argument the constant kappa 
    (which can take any value, big or small).
    """
    if distribution.lower() == "newman powerlaw":
        if len(constants) != 2:
            raise ValueError("A Newman Powerlaw distribution needs 2 constants: a cutoff and theta.")
        else: 
            cutoff = constants[0]
            theta = constants[1]
        accept = False
        while accept == False:
            random = np.random.rand()
            k = -cutoff * np.log(1-random)
            acceptance = np.random.rand()
            if acceptance < k**(-theta) and k >= 1:
                accept = True
        return k
    
    elif distribution.lower() == "geometric":
        if len(constants) != 1:
            raise ValueError("A Geometric distribution needs 1 constant: a.")
        random = np.random.rand()
        k = math.log((1 - random) / (1 - constants), constants)
        return k

    elif distribution.lower() == "exponential":
        if len(constants) != 1:
            raise ValueError("An Exponential distribution needs 1 constant: kappa.")
        else:
            kappa = constants
        accept = False
        while accept == False:
            random = np.random.rand()
            k = -kappa * np.log(1-random)
            random2 = np.random.rand()
            if random2 <= (1 - math.e**(-1/kappa)):
                accept = True
        return k
   
    else:
        raise ValueError("This is an unknown distribution. Please pick from: newmans powerlaw, geometric, exponential.")




def Percolation(graph, phi):
    """
    Returns a percolated version of a graph.
    Takes as arguments the graph you want to percolate and the probability
    that a given node stays in the network.
    """
    Percolated_graph = graph.copy(graph)
    for node in graph.nodes():
        p = np.random.rand()
        if p > phi:
            Percolated_graph.remove_node(node)
    return Percolated_graph
    

