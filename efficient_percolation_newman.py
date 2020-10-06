import networkx as nx
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import math
import percolate
plt.style.use('seaborn')

#start with a graph

#copy it

#take a node at random --> move it to the percolated graph & remove from original one
    #add all relevant edges
    #change the cluster labels
    #update 


#Below an implementation of the Newman-Ziff algorithm as made by tha package 'percolate' in python


# number of runs
runs = 1
# system sizes
number_of_nodes = [10**5,10**5]
constants = [5]
#
even = False
sequences = []
for i in number_of_nodes:
    while even == False:
        rng = np.random.default_rng()
        sequence = rng.poisson(constants[0],i)
        if (sum(sequence) % 2) == 0:
            even == True
    sequences.append(sequence)
graphs = [nx.configuration_model(i) for i in sequences]

# compute the microcanonical averages for all system sizes
microcanonical_averages = [
    percolate.microcanonical_averages(
        graph=graph, runs=runs
    )
    for graph in graphs
]
# combine microcanonical averages into one array
microcanonical_averages_arrays = [
    percolate.microcanonical_averages_arrays(avg)
    for avg in microcanonical_averages
]

print(microcanonical_averages_arrays)
