import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import collections
import mpmath
import pandas as pd
import random
from statistics import mean
plt.style.use('seaborn-colorblind')

import Everything_in_functions as fct 

#imperfect - I need to add a way to include probabilities per degree & use those


def moments(graph):
    degree_sequence = np.array([degree for node, degree in graph.degree()])  # degree sequence
    degree_sequence_squared = degree_sequence**2
    degree_sequence_cubed = degree_sequence**3
    first_moment = np.mean(degree_sequence)
    second_moment = np.mean(degree_sequence_squared)
    third_moment = np.mean(degree_sequence_cubed)

    return [first_moment, second_moment, third_moment]

def moments_from_sequence(sequence):
    degree_sequence_squared = sequence**2
    degree_sequence_cubed = sequence**3
    first_moment = np.mean(sequence)
    second_moment = np.mean(degree_sequence_squared)
    third_moment = np.mean(degree_sequence_cubed)

    return [first_moment, second_moment, third_moment]

def moments_from_frequencies(freq):
    total = sum(freq['Count'])
    if total != n:
        raise ValueError
    freq['Total deg'] = freq.apply(lambda x: x['Degree']*x['Count']/(total), axis = 1)
    freq['Total deg^2'] = freq.apply(lambda x: x['Degree']**2*x['Count']/(total), axis = 1)
    freq['Total deg^3'] = freq.apply(lambda x: x['Degree']**3*x['Count']/(total), axis = 1)
    first_moment = sum(freq['Total deg'])
    second_moment = sum(freq['Total deg^2'])
    third_moment = sum(freq['Total deg^3'])
    return [first_moment, second_moment, third_moment]

def change_degrees(deg, const, distribution, mean):
    # if distribution.lower() == "geometric":
    #     prob = (1-const)*const**deg
    # if distribution.lower() == "poisson":
    #     prob = const**deg
    # if distribution.lower() == "exponential":
    #     prob = (1 - math.e**(1/const)) * math.e**(deg/const)
    random = np.random.rand()
    prob = (1-const)*const**deg
    if random > prob:
        return deg + 1
    else:
        return deg - 1


def robustness(G, phi_c):
    s = []
    for occupation_probability in np.arange(0,1,0.025):
        if occupation_probability <= phi_c:
            s.append(0)
        else: 
            P = fct.Percolation_nodes(G,occupation_probability)
            GC = max(nx.connected_components(P), key=len)
            GC = P.subgraph(GC)
            s.append(GC.number_of_nodes()/P.number_of_nodes())
    return np.mean(s)
        
        

#Setting values for all parameters
n = 10**4
a = [0.6]
degree_distribution_choice = 'Geometric'
#change_distribution = "geometric"
multiplication_factor = 3

# Creating an even degree sequence for 
sequence = np.array(fct.degree_sequence(a, n, degree_distribution_choice))
mean = np.mean(np.array(sequence))
G = nx.configuration_model(sequence)
old_G = G
critical = fct.critical_value(G, a, degree_distribution_choice)

changed = 0
for i in range(int(G.number_of_edges()/100)):
    old_R = robustness(G, critical)
    edge1, edge2 = random.sample(list(G.edges()), k=2)
    new_edge1 = [edge1[0],edge2[1]]
    new_edge2 = [edge2[0],edge1[1]]
    G.remove_edges_from([edge1, edge2])
    G.add_edges_from([new_edge1, new_edge2])
    new_critical = fct.critical_value(G, a, degree_distribution_choice)
    new_R = robustness(G, new_critical)
    if old_R < new_R:
        old_G = G
        changed += 1 
    else: 
        G = old_G

sizes1 = []
sizes2 = []
occupation_probabilities = range(0,1,0.025)
for occupation_probability in occupation_probabilities:
        if occupation_probability <= critical:
            sizes1.append(0)
        else: 
            P = fct.Percolation_nodes(G,occupation_probability)
            GC = max(nx.connected_components(P), key=len)
            GC = P.subgraph(GC)
            sizes.append(GC.number_of_nodes()/P.number_of_nodes())

        if occupation_probability <= new_critical:
            sizes2.append(0)
        else: 
            P2 = fct.Percolation_nodes(G,occupation_probability)
            GC2 = max(nx.connected_components(P2), key=len)
            GC2 = P.subgraph(GC2)
            sizes2.append(GC2.number_of_nodes()/P2.number_of_nodes())

plt.scatter(occupation_probabilities, sizes1, label = "Before Rewiring", color = 'red')
plt.scatter(occupation_probabilities, sizes2, label = "After Rewiring", color = 'blue')

plt.scatter(critical,0,marker='o', label = "Critical Value Before", color = 'red')
plt.scatter(new_critical,0,marker = 'o', label = "Critical Value After", color = 'blue')

plt.title("Size of Giant Component and Occupation Probability")
plt.legend()
plt.xlabel("Occupation Probability ($\phi$)")
plt.ylabel("Size of Giant Component (s)")
plt.show()


# #Using the change_degrees function to change the degree with a certain probability
# new_sequence = pd.Series(sequence,name = "Degree")
# even = False
# while even == False:
#     new_sequence = new_sequence.apply(lambda x: change_degrees(x, multiplication_factor, change_distribution, mean))
#     total = sum(new_sequence)
#     if total % 2 == 0:
#         even = True
#     else:
#         print("Odd")

# #Creates series of degree sequence
# degree_sequence = pd.Series(sequence, name = "Degree")
# degree_frequencies = degree_sequence.value_counts()
# degree_frequencies.name = 'Count'

# #Make it a df of degree&count
# degree_frequencies = degree_frequencies.reset_index()
# degree_frequencies.columns = ["Degree", "Count"]
# degree_frequencies['Degree'] = degree_frequencies['Degree']*multiplication_factor**(1/3)

# #Multiplying the value counts by a
# degree_frequencies = degree_frequencies*multiplication_factor

new_G = nx.configuration_model(new_sequence)

#Calculate the moments
origmo = np.array(moments_from_sequence(sequence))
#newmo = np.array(moments_from_frequencies(degree_frequencies))
newmo = np.array(moments(new_G))
multiplied = newmo/origmo

print(origmo)
print(newmo)
print(multiplied)



# df = pd.DataFrame(differences[-1], 
#                     columns = ["First",'Second', 'Third'], 
#                     index = ["Original",'New','Factor'])

# print(df.to_latex())

#Stuff
# number_degrees = degree_sequence.value_counts(normalize = True)
# number_degrees = number_degrees*multiply
# new_total = sum(number_degrees)
# zeroes = 1 - new_total
# number_degrees[0] = zeroes
# print(number_degrees)

# # Remove edges from nodes with a certain probability
# new_sequence = degree_sequence.apply(lambda x: change_degrees(x, multiply))


# # Remove all edges from nodes with degree higher than 4
# new_sequence = pd.Series(0 if deg > 4 else deg for deg in sequence)

# #Remove all nodes with degree higher than 4
# higher_than_four = degree_sequence[degree_sequence>4]
# new_sequence = degree_sequence[degree_sequence<=4]
# edges_removed = round(sum(higher_than_four)/2)

# # Add edges to nodes with small degree.
# new_sequence = new_sequence.sort_values()
# new_sequence[:nodes_removed] = new_sequence[:edges_removed]+1


# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html


