import numpy as np
import math
import networkx as nx
import mpmath

# u = 0.1
# for i in range(100000):
#     u1 = u
#     u = 0.4**2 / (1 - 0.6*u1)**2

# g0 = (0.4) / (1 - 0.6*u)

# print(1 - g0)

print(mpmath.polylog(1.5, math.e**(-1/10)))