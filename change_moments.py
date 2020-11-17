import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import collections
import mpmath
import pandas as pd
import random
plt.style.use('seaborn')

import Everything_in_functions as fct 

#imperfect - I need to add a way to include probabilities per degree & use those


def moments(graph):
    degree_sequence = np.array([degree for node, degree in graph.degree()])  # degree sequence
    degree_sequence_squared = degree_sequence**2
    degree_sequence_cubed = degree_sequence**3
    first_moment = np.mean(degree_sequence)
    second_moment = np.mean(degree_sequence_squared)
    third_moment = np.mean(degree_sequence_cubed)

    print(first_moment, second_moment, third_moment)

    return first_moment, second_moment, third_moment

def moments_from_sequence(sequence):
    degree_sequence_squared = sequence**2
    degree_sequence_cubed = sequence**3
    first_moment = np.mean(sequence)
    second_moment = np.mean(degree_sequence_squared)
    third_moment = np.mean(degree_sequence_cubed)

    print(first_moment, second_moment, third_moment)

    return [first_moment, second_moment, third_moment]

#Setting values for all parameters
n = 1000
a = [3]
degree_distribution_choice = 'Poisson'

# Creating an even degree sequence for 
sequence = fct.degree_sequence(a, n, degree_distribution_choice)

G = nx.configuration_model(sequence)
original = G.copy()

#Creates the degree sequence including node labels
degree_sequence = []
for node, degree in G.degree():
    degree_sequence.append([node,degree])
degree_sequence = pd.DataFrame(degree_sequence, columns = ['Node', 'Degree'])

#Remove all nodes with degree higher than 4
higher_than_four = degree_sequence[degree_sequence['Degree']>4]
nodes_removed = round(sum(higher_than_four['Degree'])/2)
degree_sequence = degree_sequence.append(higher_than_four)
degree_sequence = degree_sequence.drop_duplicates(subset ='Node',keep=False)

#Add edges to nodes with small degree.
degree_sequence = degree_sequence.sort_values(by = "Degree")
degree_sequence[:nodes_removed] = degree_sequence[:nodes_removed]+1

#Create new graph with removed nodes
New = original.subgraph(list(degree_sequence['Node']))

origmo = moments(original)
newmo = moments_from_sequence(degree_sequence['Degree'])

# edges_to_add = 0
# degree_sequence = []
# diff_degrees = []
# for node, degree in G.degree():
#     if degree not in diff_degrees:
#         diff_degrees.append(degree)
#     degree_sequence.append([node, degree])
#     if degree > 4:
#         for edge in original.edges():
#             if node in edge:
#                 edges_to_add += 1
#                 G.remove_edges_from([edge])

# diff_degrees.sort()
# degree_df = pd.DataFrame(degree_sequence, columns = ["Node", "Degree"])
# degree_df = degree_df.sort_values(by = "Degree", ascending = True)
# while edges_to_add != 0:
#     for degree in diff_degrees:
#         nodes_with_degree = list(degree_df[degree_df["Degree"]==degree]['Node'])
#         while len(nodes_with_degree) >= 2:
#             node1 = random.choice(nodes_with_degree)
#             nodes_with_degree.remove(node1)
#             node2 = random.choice(nodes_with_degree)
#             nodes_with_degree.remove(node2)
#             G.add_edge(node1,node2)
#             edges_to_add -= 1





# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html