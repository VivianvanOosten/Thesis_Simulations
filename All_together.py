import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math

def dd_geometric(k, constant, phi = None):
    """
    Given a degree k, returns the probability that any random node has this degree
    according to the degree distribution p_k = (1-a)a^k.
    Takes as arguments the degree k, the constant a and optionally phi.
    If phi is included as an argument, it uses the percolated degree distribution:
    p_k = (1-a) (a phi)^(1+a(phi-1))^(-(k+1)).
    """
    if phi == None:
        probability = (1-constant) * constant**k 
        return probability
    else:
        probability = (1-constant) * (constant * phi)**k * (1 + constant*(phi-1))**(-(k+1))
        return probability

def networkx_powerlaw(n, a):
    """
    Uses the inbuilt function random_sequence.powerlaw_sequence of NetworkX
    to return a random degree sequence with even total degree.
    Takes as arguments the number of nodes in the degree sequence and the
    exponent of the powerlaw distribution.

    Problem: I don't know exactly what that distribution is.
    """
    even = False
    while even == False:
        degree_sequence = nx.utils.random_sequence.powerlaw_sequence(n, exponent = a)
        degree_sequence = [int(i) for i in degree_sequence]
        total = sum(degree_sequence)
        if (total % 2) == even:
            even = True
    return degree_sequence

def number_generator_powerlaw(cutoff, theta):
    """
    Uses the same algorithm Newman in his paper did (page 9) to return a 
    random number of the powerlaw distribution with a cutoff and an 
    exponent theta.
    Takes as arguments the cutoff value and the exponent.
    """
    accept = False
    while accept == False:
        random = np.random.rand()
        k = -cutoff * np.log(1-random)
        acceptance = np.random.rand()
        if acceptance < k**(-theta) and k >= 1:
            accept = True
    return k

def number_generator_geometric(constant):
    """
    Generates a k in the degree distribution: p_k = (1-a)a^k. 
    Found on Newman book page 580.
    The method for generating this number was taken 
    from the Newman article, page 9. 
    Takes as argument the constant a. 
    """
    random = np.random.rand()
    k = math.log((1 - random) / (1 - constant), constant)
    return k

def number_generator_exponential(kappa):
    """
    Generates a k in the degree distribution:
    p_k = (1 - e**(-1/kappa)) e**(-k/kappa).
    Found in Nemwman particle page 4.
    Takes as argument the constant kappa 
    (which can take any value, big or small)
    """
    accept = False
    while accept == False:
        random = np.random.rand()
        k = -kappa * np.log(1-random)
        random2 = np.random.rand()
        if random2 <= (1 - math.e**(-1/kappa)):
            accept = True
    return k

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

def Fig_degree_distribution_geometric(graph, constant, phi = None):
    #Find the actual and expected degree frequencies
    p_k = nx.degree_histogram(graph)
    total = graph.number_of_nodes()
    p_k = [i/total for i in p_k]
    index = range(len(p_k))
    expected = []
    for x in index:
        expected.append( dd_geometric(x, constant, phi) )

    # Plot the degree frequencies

    #Plotting degree frequencies in a log scale
    histogram = plt.figure()
    ax = histogram.add_subplot(1,1,1)
    ax.set_yscale('log')
    #ax.set_xscale('log')

    #Can pick one of the options below to plot the actual frequencies
    #plt.plot(p_k) #plots frequences as a line
    plt.scatter(x = index, y = p_k) #plots as points
    plt.plot(expected)

    return histogram


#defining the number of nodes in the network (n)
#and the relevant constants for each type of distribution
n = 10000
cutoff = 10
theta = 1
a = 0.5

# Generate n nodes in the a specific distribution
# Throw away the sequence if the sum of degrees is odd & try again until even
degree_sequence = []
even = False
while even == False:
    for i in range(n):
        # Which distribution is used can be chosen below
        degree_sequence.append(int(number_generator_geometric(a)))
    total = sum(degree_sequence)
    if (total % 2) == 0 :
        even = True
    
#Now we have a degree sequence

#Make the graph using the configuration model
G = nx.configuration_model(degree_sequence)

# # Plot the degree distribution
# degree_distribution = Fig_degree_distribution_geometric(G, a)
# plt.show()

# Plotting degree distributions for percolated graph
# Percolating the graph
occupation_probability = 0.6
P = Percolation(G,occupation_probability)
percolated_distribution = Fig_degree_distribution_geometric(P, a, occupation_probability)
plt.show()


# #Plotting the size of the Giant Component against occupation probability

# graph_data = [[],[]] # going to be: [[occupation probabilities], [sizes of GC]]

# for occupation_probability in np.arange(0.05,1.1,0.05):
#     P = Percolation(G,occupation_probability)
#     GC = max(nx.connected_components(P), key=len)
#     GC = P.subgraph(GC)
#     graph_data[0].append(occupation_probability)
#     graph_data[1].append(GC.number_of_nodes())

# plt.plot(graph_data[0],graph_data[1])
# plt.show()


