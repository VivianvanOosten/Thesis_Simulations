import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import collections
import mpmath
plt.style.use('seaborn')

# General to do's:
#   - add Newman's powerlaw to the degree distribution generators

def number_generator(constants, distribution):
    """
    Generates a number according to a specific distribution and its specified constants.
    Options for distributions are:
    - Newmans Powerlaw
    - Geometric
    - Exponential
    - Poisson

    Note that the Poisson distribution is generated using the Numpy function for it.

    Newmans Powerlaw:
    Uses the same algorithm Newman in his paper did (page 9) to return a 
    random number of the powerlaw distribution with a cutoff and an 
    exponent theta.
    Takes as arguments the cutoff value and the exponent.

    Geometric:
    Generates a k in the degree distribution: p_k = (1-a)a^k. 
    Found on Newman book page 580.
    The method for generating this number was taken 
    from the Newman article, page 9 and adapted for this distribution.
    Takes as argument the constant a. 

    Exponential:
    Generates a k in the degree distribution:
    p_k = (1 - e**(-1/kappa)) e**(-k/kappa).
    Found in Nemwman particle page 4.
    Takes as argument the constant kappa 
    (which can take any value, big or small).
    """
    if distribution.lower() == "scale free":
        if len(constants) != 2:
            raise ValueError("A scale free distribution needs 2 constants: a cutoff and tau.")
        cutoff = constants[0]
        tau = constants[1]
        accept = False
        while accept == False:
            random = np.random.rand()
            k = math.ceil(-cutoff * np.log(1-random))
            if round(k) == 0:
                continue
            acceptance = np.random.rand()
            if acceptance <= k**(-tau): #removed k>=1
                accept = True
        # while accept == True:
        #     random2 = np.random.rand()
        #     if random2 <= mpmath.polylog(tau, math.e**(-1/cutoff)):
        #         accept = False
        return k
    
    elif distribution.lower() == "geometric":
        if len(constants) != 1:
            raise ValueError("A Geometric distribution needs 1 constant: a.")
        a = constants[0]
        accept = False
        while accept == False:
            random = np.random.rand()
            k = math.log((1 - random), a)
            random2 = np.random.rand()
            if random2 <= (1-a):
                accept = True
        return math.floor(k)

    elif distribution.lower() == "exponential":
        if len(constants) != 1:
            raise ValueError("An Exponential distribution needs 1 constant: kappa.")
        kappa = constants[0]
        accept = False
        while accept == False:
            random = np.random.rand()
            k = -kappa * np.log(1-random)
            if k == 0:
                continue
            random2 = np.random.rand()
            if random2 <= (1 - math.e**(-1/kappa)):
                accept = True
        return math.floor(k)

    # Early version of the scale free variant
    # elif distribution.lower() == "scale free":
    #     if len(constants) != 1:
    #         raise ValueError("A scale-free distribution needs 1 constant: gamma.")
    #     gamma = constants[0]
    #     random = np.random.rand()
    #     k = random**(-1/gamma)
    #     return math.floor(k)


    #Poisson distribution is generated in one go using the numpy poisson generation 

    else:
        raise ValueError("This is an unknown distribution. Please pick from: newmans powerlaw, geometric, exponential, scale-free.")


def degree_distribution(k, constants, distribution, phi = None):
    """
    For a given k, it returns the probability that any 
    random node has this degree according to a given distribution.
    The possible distributions are:
    - geometric
    - exponential
    - Poisson

    Arguments are: k, constants, the distribution and optionally phi. 

    Geometric:
    The distribution is: p_k = (1-a)a^k.
    If phi is included as an argument, it uses the percolated degree distribution:
    p_k = (1-a) (a phi)^(1+a(phi-1))^(-(k+1)).

    Exponential:
    The distribution is: p_k = (1 - e^(1/kappa) ) * e^(-k/kappa)).
    if phi is included as an argument, it uses the percolated degree distribution:
    p_k = ?????

    Poisson:
    The distribution is: p_k = e^(-c) c^k / k!.
    If phi is included as an argument, it uses the percolated degree distribution:
    p_k = e^(-c phi) (c phi)^k / k!
    """
    #To be done: add newman's powerlaw distribution here.

    if distribution.lower() == 'geometric':
        constant = constants[0]
        if phi is None:
            probability = (1-constant) * constant**k 
        else:
            probability = (1-constant) * ((constant*phi)**k) / ((1 + constant*phi - constant)**(k+1))
        return probability
    
    if distribution.lower() == 'exponential':
        constant = constants[0]
        if phi is None:
            probability = (1 - math.e**(-1/constant)) * math.e**(-k/constant)
        else:
            probability = (1 - math.e**(-1/constant)) * math.e**(-k/constant) * phi**k / ( 1 + (phi-1)*math.e**(-1/constant))**(k+1)
        return probability
    
    if distribution.lower() == 'poisson':
        constant = constants[0]
        if phi is None:
            probability = math.e**(-constant) * constant**k / math.factorial(k)
        else:
            probability = math.e**(-constant*phi) * (constant*phi)**k / math.factorial(k)
        return probability

    if distribution.lower() == 'scale free':
        cutoff = constants[0]
        tau = constants[1]
        if k == 0:
            probability == 0
        elif phi is None:
            probability = k**(-tau) * math.e**(-k/cutoff) / mpmath.polylog(tau,math.e**(-1/cutoff))
        else:
            probability = 0
        return float(probability)



def degree_sequence(constants, number_of_nodes, distribution):
    """
    Creates a degree sequence with the given specifications:
    constants, a number of nodes and the required distribution.
    For Poisson it generataes the sequence directly. 
    All other distributions are generated using the 'number_generator' fct.
    """
    even = False
    while even == False:
        sequence = []
        if distribution.lower() == 'poisson':
            rng = np.random.default_rng()
            sequence = rng.poisson(constants[0],number_of_nodes)
        else:
            for i in range(number_of_nodes):
                # Which distribution is used can be chosen below
                sequence.append(round(number_generator(constants, distribution)))
        total = sum(sequence)
        if (total % 2) == 0:
            even = True
    return sequence


def Percolation_nodes(graph, phi):
    """
    Returns a percolated version of a graph.
    Takes as arguments the graph you want to percolate and the probability
    that a given node stays in the network.
    """
    Percolated_graph = graph.copy(graph)
    for node in graph.nodes():
        p = np.random.rand()
        if p >= phi:
            Percolated_graph.remove_node(node)
    return Percolated_graph


def Percolation_edges(graph, phi):
    """
    Returns a percolated version of a graph.
    Takes as arguments the graph you want to percolate and the probability
    that a given edge stays in the network.
    """
    Percolated_graph = graph.copy(graph)
    for u,v in graph.edges():
        p = np.random.rand()
        if p >= phi:
            Percolated_graph.remove_edge(u,v)
    return Percolated_graph


def Fig_degree_distribution(graph, constants, distribution, phi = None):
    """
    Draws the degree distribution graph, both simulated and theorized. 
    It does this for the following degree distributions:
    """
    degree_sequence = sorted([d for n, d in graph.degree()], reverse=True)  # degree sequence
    degreeCount = collections.Counter(degree_sequence)
    deg, cnt = zip(*degreeCount.items())
    total_nodes = sum(cnt)
    fractions = [i/total_nodes for i in cnt]
    norm = sum(fractions)
    # if norm != 1:
    #     raise ValueError("Simulation incorrectly normalised, is now {}".format(norm))
    
    expected = []
    for x in deg:
        expected.append( degree_distribution(x, constants, distribution, phi) )
    total_expected = sum(expected)
    expected = [i/total_expected for i in expected]
    norm = sum(expected)
    print(norm)
    # if norm != 1:
    #     raise ValueError("Theory incorrectly normalised, is now {}".format(norm))

    # Plot the degree frequencies

    #Plotting degree frequencies in a log scale
    histogram = plt.figure()
    ax = histogram.add_subplot(1,1,1)
    ax.set_yscale('log') # to make the end-result a straight line

    #Plotting everything we need
    plt.plot(deg, expected, label = "Theory, c = {}".format(constants[0]))
    plt.scatter(deg, fractions, facecolor = 'none', edgecolors = 'black', label = "Simulation, c = {}".format(constants[0]))

    return None


def critical_value(G, constants, distribution):
    """
    Calculates the critical value of phi.
    For any known distribution we can easily calculate it,
    using the analytical shortcuts.
    For any unknown or previously uncalculated distribution,
    we use the moments of the graph.
    """

    if distribution.lower() == "poisson":
        return 1 / constants[0]
    
    if distribution.lower() == "geometric":
        constant = constants[0]
        return (1-constant) / (2*constant)

    else:
        degree_sequence = np.array([degree for node, degree in G.degree()])
        degree_sequence_squared = np.array([degree**2 for degree in degree_sequence])
        phi_c = np.mean(degree_sequence) / (np.mean(degree_sequence_squared) - np.mean(degree_sequence))
        return phi_c


def tangent_by_moments(G,phi):
    degree_sequence = np.array([degree for node, degree in G.degree()])  # degree sequence
    degree_sequence_squared = degree_sequence**2
    degree_sequence_cubed = degree_sequence**3
    first_moment = np.mean(degree_sequence)
    second_moment = np.mean(degree_sequence_squared)
    third_moment = np.mean(degree_sequence_cubed)
    numerator = second_moment - first_moment
    denominator =  third_moment - (3*second_moment) + (2*first_moment) 
    return 2 * numerator**2 / denominator


def tangent_poisson(constants,phic):
    constant = constants[0]
    return constant / (constant**2 * phic**2)


def tangent_geometric(constants,phic):
    constant = constants[0]
    return 4/(3*phic)




