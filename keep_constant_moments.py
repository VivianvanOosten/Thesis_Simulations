import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import collections
import mpmath
import random
from statistics import mean
plt.style.use('seaborn')

import Everything_in_functions as fct 

def moments_from_sequence(sequence):
    degree_sequence_squared = sequence**2
    degree_sequence_cubed = sequence**3
    first_moment = np.mean(sequence)
    second_moment = np.mean(degree_sequence_squared)
    third_moment = np.mean(degree_sequence_cubed)

    return [first_moment, second_moment, third_moment]

def moments_from_graph(graph):
    degree_sequence = np.array([degree for node, degree in graph.degree()])
    return moments_from_sequence(sequence)


#Setting values for all parameters
n = 10**4
a = [0.6]
degree_distribution_choice = 'Geometric'
#change_distribution = "geometric"
multiplication_factor = 3


sequence = np.array(fct.degree_sequence(a, n, degree_distribution_choice))
G_original = nx.configuration_model(sequence)
G = G_original
old_G = G
changed = 0

for i in range(100):
    edge1 = random.choice(list(G.edges()))
    edge2 = random.choice(list(G.edges()))
    new_edge1 = [edge1[0],edge2[1]]
    new_edge2 = [edge2[0],edge1[1]]
    G.remove_edges_from([edge1, edge2])
    G.add_edges_from([new_edge1, new_edge2])
    moments_new = moments_from_graph(G)
    moments_old = moments_from_graph(old_G)
    if moments_old[0:2] == moments_new[0:2]:
        old_G = G
        changed += 1 
    else: 
        G = old_G

print(changed)
print(moments_from_graph(G))
print(moments_from_graph(G_original))

phi_c = fct.critical_value(G, a, degree_distribution_choice)

sizes_original = []
sizes_new = []
sizes_new = []
theory_u = []
occupation_probabilities = np.arange(0, 1.05, 0.05)

for occupation_probability in occupation_probabilities:
    if occupation_probability <= phi_c:
        sizes_original.append(0)
        sizes_new.append(0)
        theory_u.append(0)
        continue
    iterative_size_new = []
    iterative_size_old = []
    P_or = fct.Percolation_nodes(G_original,occupation_probability)
    P_new = fct.Percolation_nodes(G,occupation_probability)
    GC_or = max(nx.connected_components(P_or), key=len)
    GC_or = P_or.subgraph(GC_or)
    GC_new = max(nx.connected_components(P_new), key=len)
    GC_new = P_new.subgraph(GC_new)
    iterative_size_new.append(GC_new.number_of_nodes()/P_new.number_of_nodes())
    iterative_size_old.append(GC_or.number_of_nodes()/P_or.number_of_nodes())
    sizes_original.append(mean(iterative_size_old))
    sizes_new.append(mean(iterative_size_new))


plt.plot(occupation_probabilities, sizes_original, label = "Original network")
plt.plot(occupation_probabilities, sizes_new, label = "Rewired network")
plt.scatter(phi_c,0,marker='o', label = "Critical Value")

# plt.plot(occupation_probabilities, theory_u, label = "Theorized size")

# tangent_plot = np.arange(phi_c-0.05, phi_c+0.2, 0.05)
# tangent = fct.tangent_by_moments(G, phi_c)

# plt.plot(tangent_plot, ((tangent_plot * tangent) - (phi_c * tangent)), label = "Tangent")

plt.title("Size of Giant Component and Occupation Probability")

plt.legend()
plt.xlabel("Occupation Probability ($\phi$)")
plt.ylabel("Size of Giant Component (s)")
plt.show()




