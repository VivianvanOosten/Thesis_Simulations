import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import collections
import mpmath
plt.style.use('seaborn')

import Everything_in_functions as fct 


#Setting values for all parameters
n = 10**5
a = [5]
degree_distribution_choice = 'Poisson'
# constants_options = [[3],[5],[8]]

# for a in constants_options:
# Creating an even degree sequence for 
sequence = fct.degree_sequence(a, n, degree_distribution_choice)

G = nx.configuration_model(sequence)


#Plotting percolated degree distributions
occupation_probability = 0.6
P = fct.Percolation_edges(G,occupation_probability)
fct.Fig_degree_distribution(P, a, degree_distribution_choice, occupation_probability)

#Plotting the size of the Giant Component against occupation probability   

plt.xlabel("Degree (k)")
plt.ylabel("Degree probability ($p_k$)")
plt.legend()
plt.title("Degree Distribution for {} distribution".format(degree_distribution_choice))   
plt.show()