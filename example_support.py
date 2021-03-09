import numpy as np
import random as rn

parent_1 = np.array([True,True,False,False,True],
                    [True,False,False,False,True],
                    [True,True,False,True,True])
ind = np.array([True,True,False,False,False])

print(parent_1[0] & ind)