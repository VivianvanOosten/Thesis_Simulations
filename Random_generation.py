import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

cutoff = 10
theta = 1

def random_powerlaw_cutoff(cutoff, theta):
    """
    Uses the same algorithm Newman in his paper did (page 9) to return a 
    random number of the powerlaw distribution with a cutoff and an 
    exponent theta.
    Takes as arguments the cutoff value and the exponent.
    """
    accept = False
    while accept == False:
        random = np.random.rand()
        k = -cutoff * np.log(1-random)
        acceptance = np.random.rand()
        if acceptance < k**(-theta) and k >= 1:
            accept = True
    return k

# Generate n nodes in the powerlaw distribution
# Throw away the sequence if the sum of degrees is odd & try again until even
n = 10
degree_sequence = []
even = False
while even == False:
    for i in range(n):
        degree_sequence.append(int(random_powerlaw_cutoff(cutoff,theta)))
    total = sum(degree_sequence)
    if (total % 2) == 0 :
        even = True

#Now we have a degree sequence.    