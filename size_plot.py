import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import collections
import mpmath
plt.style.use('seaborn')

import Everything_in_functions as fct 


#Setting values for all parameters
n = 10**2
a = [3]
degree_distribution_choice = 'Exponential'
# constants_options = [[3],[5],[8]]

# for a in constants_options:
# Creating an even degree sequence for 
sequence = fct.degree_sequence(a, n, degree_distribution_choice)

G = nx.configuration_model(sequence)

#Plotting the size and phi in one graph
phi_c = fct.critical_value(G, a, degree_distribution_choice)
#for a zoomed in graph
#occupation_probabilities = np.arange(phi_c - 0.1, phi_c + 0.1,0.005)
#for a full graph
sizes = []
occupation_probabilities = np.arange(0, 1.05, 0.05)

for occupation_probability in occupation_probabilities:
    if occupation_probability <= phi_c:
        sizes.append(0)
        continue
    P = fct.Percolation_nodes(G,occupation_probability)
    GC = max(nx.connected_components(P), key=len)
    GC = P.subgraph(GC)
    sizes.append(GC.number_of_nodes()/P.number_of_nodes())

plt.plot(occupation_probabilities, sizes, label = "{} distribution, with c = {}".format(degree_distribution_choice, a[0]))
plt.scatter(phi_c,0,marker='o', label = "Critical Value")


plt.title("Size of Giant Component and Occupation Probability")

#Plotting the tangent line for: 
tangent_plot = np.arange(phi_c-0.05, phi_c+0.2, 0.05)
tangent = fct.tangent_by_moments(G, phi_c)

plt.plot(tangent_plot, ((tangent_plot * tangent) - (phi_c * tangent)), label = "Tangent")

plt.legend()
plt.xlabel("Occupation Probability ($\phi$)")
plt.ylabel("Size of Giant Component (s)")
plt.show()
