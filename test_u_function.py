import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import collections
import mpmath
plt.style.use('seaborn')

import Everything_in_functions as fct 


def u_finder(distribution, constants, phi):
    u = 0.5
    u_1 = 0
    constant = constants[0]
    if distribution.lower() == 'geometric':
        for i in range(10000):
            u_1 = u
            numerator = 1 - constant
            denominator = 1 - (constant*(u_1*phi - phi + 1))
            u = (numerator**2) / (denominator**2)
        return u

    if distribution.lower() == 'poisson':
        for i in range(10000):
            u_1 = u
            u = math.e**(constant*(u_1*phi - phi))
        return u


def s_finder(distribution, constants, phi):
    s = 0.5
    i = 0
    s_1 = 0
    constant = constants[0]
    if distribution.lower() == 'geometric':
        while abs(s - s_1) > 0.01:
            s_1 = s
            numerator = 1 - constant
            denominator = 1 - (constant*(1- (s*phi)))
            s = 1 - numerator**2 / denominator**2
        return s

    if distribution.lower() == 'poisson':
        while abs(s - s_1) > 0.1:
            s_1 = s
            s = 1 - math.e**(constant * phi * s)
        return s



#Setting values for all parameters
n = 10**5
a = [3]
degree_distribution_choice = 'Poisson'

# Creating an even degree sequence for 
sequence = fct.degree_sequence(a, n, degree_distribution_choice)

G = nx.configuration_model(sequence)

#Plotting the size and phi in one graph
phi_c = fct.critical_value(G, a, degree_distribution_choice)

sizes = []
theory_u = []
occupation_probabilities = np.arange(0, 1.05, 0.0025)

for occupation_probability in occupation_probabilities:
    if occupation_probability <= phi_c:
        sizes.append(0)
        theory_u.append(0)
        continue
    iterative_size = []
    for i in range(1000):
        P = fct.Percolation_nodes(G,occupation_probability)
        GC = max(nx.connected_components(P), key=len)
        GC = P.subgraph(GC)
        iterative_size.append(GC.number_of_nodes()/P.number_of_nodes())
    sizes.append(mean(iterative_size))


    # ADD THEORETICAL S HERE

    u = u_finder(degree_distribution_choice, a, occupation_probability)
    if degree_distribution_choice.lower() == "geometric":
        s = occupation_probability * ( 1 - ( (1 - a[0]) / ( 1 - a[0]*(u*occupation_probability - occupation_probability + 1) ) ))
    if degree_distribution_choice.lower() == "poisson":
        s = occupation_probability * ( 1 - math.e**(a[0]*(u*occupation_probability - occupation_probability) ) )
    #s = s_finder(degree_distribution_choice, a, occupation_probability)

    #S according to Newman
    #s = (3/2)*occupation_probability - math.sqrt((1/4)*occupation_probability**2 + occupation_probability*((1/a[0]) - 1))
    theory_u.append(s)

theorized_u = plt.figure()

plt.scatter(occupation_probabilities, sizes, facecolors = 'none', label = "{} distribution, with c = {}".format(degree_distribution_choice, a[0]))
plt.scatter(phi_c,0,marker='o', label = "Critical Value")

plt.plot(occupation_probabilities, theory_u, label = "Theorized size")

tangent_plot = np.arange(phi_c-0.05, phi_c+0.2, 0.05)
tangent = fct.tangent_by_moments(G, phi_c)

plt.plot(tangent_plot, ((tangent_plot * tangent) - (phi_c * tangent)), label = "Tangent")

plt.title("Size of Giant Component and Occupation Probability")


plt.legend()
plt.xlabel("Occupation Probability ($\phi$)")
plt.ylabel("Size of Giant Component (s)")
plt.savefig("{}_node_{}.png".format(degree_distribution_choice, a[0]))
plt.show()
