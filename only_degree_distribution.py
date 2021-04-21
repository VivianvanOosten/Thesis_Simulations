import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import collections
import mpmath
plt.style.use('seaborn')

import Everything_in_functions as fct 


#Setting values for all parameters
n = 10**6
a = [100, 2.5]
degree_distribution_choice = 'scale free'
# constants_options = [[3],[5],[8]]

# for a in constants_options:
# Creating an even degree sequence for 

#G = nx.configuration_model(sequence)


#Plotting percolated degree distributions
occupation_probability = None
#P = fct.Percolation_edges(G,occupation_probability)
#fct.Fig_degree_distribution(G, a, degree_distribution_choice, occupation_probability)
#for iteration in range(100):
sequence = fct.degree_sequence(a, n, degree_distribution_choice)
sequence.sort()
degreeCount = collections.Counter(sequence)
deg, cnt = zip(*degreeCount.items())
fractions = [i/n for i in cnt]
norm = sum(fractions)
print(norm)
# if norm != 1:
#     raise ValueError("Simulation incorrectly normalised, is now {}".format(norm))

expected = []
for x in deg:
    expected.append(fct.degree_distribution(x, a, degree_distribution_choice, occupation_probability) )
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
plt.plot(deg, expected, label = "Theory")
plt.scatter(deg, fractions, facecolor = 'none', edgecolors = 'black', label = "Simulation")

#Plotting the size of the Giant Component against occupation probability   

plt.xlabel("Degree (k)")
plt.ylabel("Degree probability ($p_k$)")
plt.legend()
plt.title("Degree Distribution for {} distribution".format(degree_distribution_choice))   
plt.show()

