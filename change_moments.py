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

    return [first_moment, second_moment, third_moment]

def moments_from_sequence(sequence):
    degree_sequence_squared = sequence**2
    degree_sequence_cubed = sequence**3
    first_moment = np.mean(sequence)
    second_moment = np.mean(degree_sequence_squared)
    third_moment = np.mean(degree_sequence_cubed)

    return [first_moment, second_moment, third_moment]

def moments_from_frequencies(freq):
    total = sum(freq)
    freq = freq.reset_index()
    print(freq)
    freq['Total deg'] = freq.apply(lambda x: x['index']*x['Count']/(total), axis = 1)
    freq['Total deg^2'] = freq.apply(lambda x: x['index']**2*x['Count']/(total), axis = 1)
    freq['Total deg^3'] = freq.apply(lambda x: x['index']**3*x['Count']/(total), axis = 1)
    first_moment = sum(freq['Total deg'])
    second_moment = sum(freq['Total deg^2'])
    third_moment = sum(freq['Total deg^3'])
    return [first_moment, second_moment, third_moment]

def change_degrees(deg, const, distribution, mean):
    if distribution.lower() == "geometric":
        prob = (1-const)*const**deg
    if distribution.lower() == "poisson":
        prob = const**deg
    if distribution.lower() == "exponential":
        prob = (1 - math.e**(1/const)) * math.e**(deg/const)
    random = np.random.rand()
    if random < prob:
        return mean*2
    else:
        return deg


#Setting values for all parameters
n = 1000000
a = [5]
degree_distribution_choice = 'Poisson'
change_distribution = "geometric"
multiplication_factor = 3

# Creating an even degree sequence for 
sequence = fct.degree_sequence(a, n, degree_distribution_choice)
mean = np.mean(np.array(sequence))

#Creates series of degree sequence
degree_sequence = pd.Series(sequence, name = "Degree")
degree_frequencies = degree_sequence.value_counts()
degree_frequencies.name = 'Count'

#Multiplying the value counts by a
degree_frequencies = degree_frequencies*multiplication_factor

#Calculate the moments
origmo = np.array(moments_from_sequence(sequence))
newmo = np.array(moments_from_frequencies(degree_frequencies))
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


