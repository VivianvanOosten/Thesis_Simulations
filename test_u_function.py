import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import collections
import mpmath
from statistics import mean
plt.style.use('seaborn')

import Everything_in_functions as fct 


def u_finder(distribution, constants, phi):
    u = 0.5
    u_1 = 0.5
    constant1 = constants[0]
    if len(constants) == 2:
        constant2 = constants[1]
    if distribution.lower() == 'geometric':
        for i in range(500):
            u_1 = u
            numerator = 1 - constant1
            denominator = 1 - (constant1*(u_1*phi - phi + 1))
            u = (numerator**2) / (denominator**2)
        return u

    if distribution.lower() == 'poisson':
        for i in range(500):
            u_1 = u
            u = math.e**(constant1*(u_1*phi - phi))
        return u
    
    if distribution.lower() == "scale free":
        for i in range(500):
            u_1 = u
            numerator = float(mpmath.polylog((constant2 - 1), ((phi*u_1 - phi + 1) * math.e**(-1/constant1))))
            denominator = (phi*u_1 - phi + 1) * float(mpmath.polylog(constant2 - 1, math.e**(-1/constant1)))
            u = numerator / denominator 
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

    if distribution.lower() == 'scale free':
        return s


#Setting values for all parameters
n = 10**6
#For powerlaw: kappa, theta
#Gotta make another one with exponent 2.5
a = [100, 2.5]
degree_distribution_choice = 'scale free'
perc_type = 'edge'

# Creating an even degree sequence for 
sequence = fct.degree_sequence(a, n, degree_distribution_choice)

G = nx.configuration_model(sequence)

#Plotting the size and phi in one graph
phi_c = fct.critical_value(G, a, degree_distribution_choice)
print(phi_c)

sizes = []
theory_u = []
differences = []
occupation_probabilities = np.arange(0, 1.05, 0.05)

for occupation_probability in occupation_probabilities:
    if occupation_probability <= phi_c:
        sizes.append(0)
        theory_u.append(0)
        print("A zero appended, {} more to go.".format(1-occupation_probability))
        continue
    iterative_size = []
    for i in range(1):
        if perc_type == 'edge':
            P = fct.Percolation_edges(G, occupation_probability)
        if perc_type == 'node':
            P = fct.Percolation_nodes(G,occupation_probability)
        GC = max(nx.connected_components(P), key=len)
        GC = P.subgraph(GC)
        iterative_size.append(GC.number_of_nodes()/n)
    to_attach = mean(iterative_size)
    sizes.append(to_attach)

    print("One more appended, {} more to go".format(1-occupation_probability))

    # # ADD THEORETICAL S HERE - nah

    u = u_finder(degree_distribution_choice, a, occupation_probability)
    
    if degree_distribution_choice.lower() == "geometric":
        if perc_type == 'edge':
            s = 1 - ( (1 - a[0]) / ( 1 - a[0]*(u*occupation_probability - occupation_probability + 1) ))
        if perc_type == 'node':
            s = occupation_probability * ( 1 - ( (1 - a[0]) / ( 1 - a[0]*(u*occupation_probability - occupation_probability + 1) ) ))
    if degree_distribution_choice.lower() == "poisson":
        if perc_type == 'edge':
            s = 1 - math.e**(a[0]*(u*occupation_probability - occupation_probability) ) 
        if perc_type == 'node':
            s = occupation_probability * ( 1 - math.e**(a[0]*(u*occupation_probability - occupation_probability) ) )
    if degree_distribution_choice.lower() == "scale free":
        to_take_log = (occupation_probability*u - occupation_probability + 1) * math.e**(-1/a[0])
        numerator = mpmath.polylog(a[1], to_take_log)
        denominator = mpmath.polylog(a[1], math.e**(-1/a[0]))
        if perc_type == 'edge': 
            s = float((1 - numerator/denominator))
        if perc_type == 'node':
            s = float(occupation_probability * ( 1 - float(numerator/denominator)))
    
    differences.append(to_attach - s)
    # #s = s_finder(degree_distribution_choice, a, occupation_probability)

    # #S according to Newman
    # #s = (3/2)*occupation_probability - math.sqrt((1/4)*occupation_probability**2 + occupation_probability*((1/a[0]) - 1))

    theory_u.append(s)

print(mean(differences))
print(sum(differences))
with open("differences_{}_{}_{}".format(degree_distribution_choice,a[0], perc_type), "w") as outfile:
    outfile.write("\n".join(str(differences)))

theorized_u = plt.figure()

if degree_distribution_choice.lower() == 'poisson':
    plt.scatter(occupation_probabilities, sizes, facecolor = 'none', edgecolors = 'black', label = "{} distribution, with c = {}".format(degree_distribution_choice, a[0]))
if degree_distribution_choice.lower() == 'geometric':
    plt.scatter(occupation_probabilities, sizes, facecolor = 'none', edgecolors = 'black', label = "{} distribution, with a = {}".format(degree_distribution_choice, a[0]))
if degree_distribution_choice.lower() == 'scale free':
    plt.scatter(occupation_probabilities, sizes, facecolor = 'none', edgecolors = 'black', label = "{} distribution, with $\kappa$ = {}, $\\tau$ = {}".format(degree_distribution_choice,a[0],a[1]))

plt.scatter(phi_c,0,marker='o', label = "Critical Value")

plt.plot(occupation_probabilities, theory_u, label = "Theorized size")

tangent_plot = np.arange(phi_c-0.05, phi_c+0.2, 0.05)
if perc_type == 'edge':
    tangent = fct.tangent_by_moments(G, phi_c)
if perc_type == 'node':
    tangent = phi_c * fct.tangent_by_moments(G, phi_c)

plt.plot(tangent_plot, ((tangent_plot * tangent) - (phi_c * tangent)), label = "Tangent")

plt.title("Size of Giant Component and Occupation Probability")


plt.legend()
plt.xlabel("Occupation Probability ($\phi$)")
plt.ylabel("Size of Giant Component (s)")
plt.show()

# if perc_type =='node':
#     plt.savefig("{}_node_{}_18/12.png".format(degree_distribution_choice, a[0]))
# if perc_type == 'edge':
#     plt.savefig("{}_edge_{}_18/12.png".format(degree_distribution_choice, a[0]))


