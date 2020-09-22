import numpy as np
import math

mini = float('inf')
maxi = float('-inf')

for kappa in np.arange(1,100,0.5):
    chance = (1-math.e**(-1/kappa))
    if mini > chance:
        mini = chance
    if maxi < chance:
        maxi = chance

print(mini,maxi)