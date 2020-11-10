import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
import math
import collections
import mpmath
plt.style.use('seaborn')

import Everything_in_functions as fct 


def u_for_threes(phi):
    # root = 4*phi**2 - 2*phi + 1
    # fraction1 = (1 - math.sqrt(root)) / (2*phi**2)
    # #fraction2= (1 - math.sqrt(root)) / (2*phi**2)
    # u1 = 1 - 1/phi + fraction1
    # #u2 = 1 - 1/phi + fraction2
    u = (2 - 2*phi) / (2*phi)
    return u#, u2

sequence = np.ones(10**5, dtype = int)
sequence = 3*sequence

G = nx.configuration_model(sequence)


phi_c = 0.5

sizes = []
theory_u_1 = []
#theory_u_2 = []
occupation_probabilities = np.arange(0, 1.05, 0.05)

for occupation_probability in occupation_probabilities:
    if occupation_probability <= phi_c:
        sizes.append(0)
        theory_u_1.append(0)
        #theory_u_2.append(0)
        continue
    P = fct.Percolation_nodes(G,occupation_probability)
    GC = max(nx.connected_components(P), key=len)
    GC = P.subgraph(GC)
    sizes.append(nx.number_of_nodes(GC)/nx.number_of_nodes(P))
    #u = u_for_threes(occupation_probability)
    #s = s_finder(degree_distribution_choice, a, occupation_probability)
    s = 2 - (1 / occupation_probability**3) + (3/occupation_probability**2) - (3/occupation_probability)
    theory_u_1.append(s)
    #theory_u_2.append(1 - u[1])

theorized_u = plt.figure()

plt.plot(occupation_probabilities, sizes, label = "Simulated size")
plt.scatter(phi_c,0,marker='o', label = "Critical Value")

plt.plot(occupation_probabilities, theory_u_1, label = "Theorized size")
#plt.plot(occupation_probabilities, theory_u_2, label = "Theorized size 2")

tangent_plot = np.arange(phi_c-0.05, phi_c+0.2, 0.05)
tangent =  (3/phi_c**4) - (6/phi_c**3) + (3/phi_c**2)

plt.plot(tangent_plot, ((tangent_plot * tangent) - (phi_c * tangent)), label = "Tangent")

plt.title("Size of Giant Component and Occupation Probability for a fixed network of degree 3")


plt.legend()
plt.xlabel("Occupation Probability ($\phi$)")
plt.ylabel("Size of Giant Component (s)")
plt.show()