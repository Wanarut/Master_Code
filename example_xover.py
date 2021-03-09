import numpy as np
import random as rn

def singlepoint_xover(A, B, x):
    A_new = np.append(A[:x], B[x:])
    B_new = np.append(B[:x], A[x:])
    return A_new, B_new

parent_1 = np.array([1,1,1,1,1,1,1,1,1])
parent_2 = np.array([2,2,2,2,2,2,2,2,2])

# Traditional crossover
# Get length of chromosome
chromosome_length = len(parent_1)

# Pick crossover point, avoding ends of chromsome
xover_points = [rn.randint(1, chromosome_length-2), rn.randint(1, chromosome_length-2)]

# Create children.
child_1 = parent_1.copy()
child_2 = parent_2.copy()
for point in xover_points:
    child_1, child_2 = singlepoint_xover(child_1, child_2, point)
print(xover_points)
print(child_1)
print(child_2)
