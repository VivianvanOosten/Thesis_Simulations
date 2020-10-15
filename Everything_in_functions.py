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
        cutoff = constants[0]
        theta = constants[1]
        accept = False
        while accept == False:
            random = np.random.rand()
            k = -cutoff * np.log(1-random)
            acceptance = np.random.rand()
            if acceptance < k**(-theta): #removed k>=1
                accept = True
        return k
    
    elif distribution.lower() == "geometric":
        if len(constants) != 1:
            raise ValueError("A Geometric distribution needs 1 constant: a.")
        random = np.random.rand()
        a = constants[0]
        k = math.log((1 - random) / (1 - a), a)
        return k

    elif distribution.lower() == "exponential":
        if len(constants) != 1:
            raise ValueError("An Exponential distribution needs 1 constant: kappa.")
        kappa = constants[0]
        accept = False
        while accept == False:
            random = np.random.rand()
            k = -kappa * np.log(1-random)
            random2 = np.random.rand()
            if random2 <= (1 - math.e**(-1/kappa)):
                accept = True
        return k

    #Poisson distribution is generated in one go using the numpy poisson generation 

    else:
        raise ValueError("This is an unknown distribution. Please pick from: newmans powerlaw, geometric, exponential.")


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
    The distribution is: p_k = (1 - e^(1-kappa) * e^(-k/kappa)).
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
            probability = (1-constant) * ((constant*phi)**k) * ((1 + constant*phi - constant)**(-k-1))
        return probability
    
    if distribution.lower() == 'exponential':
        constant = constants[0]
        if phi is None:
            probability = (1 - math.e**(-1/constant)) * math.e**(-k/constant)
        else:
            probability = None #Add whatever the phi degree distribution is for exponential functions 
        return probability
    
    if distribution.lower() == 'poisson':
        constant = constants[0]
        if phi is None:
            probability = math.e**(-constant) * constant**k / math.factorial(k)
        else:
            probability = math.e**(-constant*phi) * (constant*phi)**k / math.factorial(k)
        return probability


def degree_sequence(constants, number_of_nodes, distribution):
    """
    Creates a degree sequence with the given specifications:
    constants, a number of nodes and the required distribution.
    For Poisson it generataes the sequence directly. 
    All other distributions are generated using the 'number_generator' fct.
    """
    sequence = []
    even = False
    while even == False:
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
        if p > phi:
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
        if p > phi:
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
    plt.plot(deg, expected, label = "Theory, c = {}".format(a[0]))
    plt.plot(deg, fractions, label = "Simulation, c = {}".format(a[0]))

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
    degree_sequence_squared = np.array([degree**2 for degree in degree_sequence])
    degree_sequence_cubed = np.array([degree**3 for degree in degree_sequence])
    first_moment = np.mean(degree_sequence)
    second_moment = np.mean(degree_sequence_squared)
    third_moment = np.mean(degree_sequence_cubed)
    numerator = second_moment - first_moment
    denominator =  ( third_moment - (3*second_moment) + (2*first_moment)) * phi**2
    return numerator / denominator



#Setting values for all parameters
n = 10**5
a = [5]
degree_distribution_choice = 'Poisson'
# constants_options = [[3],[5],[8]]

# for a in constants_options:
# Creating an even degree sequence for 
sequence = degree_sequence(a, n, degree_distribution_choice)

G = nx.configuration_model(sequence)

#Plotting the size and phi in one graph
phi_c = critical_value(G, a, degree_distribution_choice)
#for a zoomed in graph
#occupation_probabilities = np.arange(phi_c - 0.1, phi_c + 0.1,0.005)
#for a full graph
sizes = []
occupation_probabilities = np.arange(0, 1.05, 0.05)

for occupation_probability in occupation_probabilities:
    if occupation_probability <= phi_c:
        sizes.append(0)
        continue
    P = Percolation_edges(G,occupation_probability)
    GC = max(nx.connected_components(P), key=len)
    GC = P.subgraph(GC)
    sizes.append(GC.number_of_nodes()/P.number_of_nodes())

plt.plot(occupation_probabilities, sizes, label = "{} distribution, with c = {}".format(degree_distribution_choice, a[0]))
plt.scatter(phi_c,0,marker='o', label = "Critical Value")


plt.title("Size of Giant Component and Occupation Probability")

#Plotting the tangent line for: 
tangent_plot = np.arange(phi_c-0.05, phi_c+0.2, 0.05)
tangent = tangent_by_moments(G, phi_c)

plt.plot(tangent_plot, (tangent_plot * tangent) - (phi_c * tangent), label = "Tangent")

plt.legend()
plt.xlabel("Occupation Probability ($\phi$)")
plt.ylabel("Size of Giant Component (s)")
plt.show()

