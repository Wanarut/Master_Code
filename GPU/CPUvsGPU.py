# conda install numba & conda install cudatoolkit

from numba import vectorize, jit
import numpy as np
# to measure exec time
from timeit import default_timer as timer

# normal function to run on cpu
def addByCPU(a):
    for i in range(a.size):
        a[i] += 1
    return a
    # return a+1

# function optimized to run on gpu 
@vectorize(['float64(float64)'], target ="cuda")
def addByGPU_vectorize(a):
    return a + 1

# @vectorize is used to write an expression that can be applied one element at a time (scalars) to an array. 
# The @jit decorator is more general and can work on any type of calculation.

@jit(nopython=True)                    
def addByGPU_jit(a):
    for i in range(a.size):
        a[i] += 1
    return a

if __name__=="__main__":
    n = 10000000
    n = 10
    a = np.ones(n, dtype = np.float64)
    b = np.ones(n, dtype = np.float64)
    c = np.ones(n, dtype = np.float64)

    if (n==10): print(a)
    start = timer()
    a = addByCPU(a)
    print("without GPU:", timer()-start)
    if (n==10): print(a)

    print('-------------vs-------------')

    if (n==10): print(b)
    start = timer()
    b = addByGPU_vectorize(b)
    print("with GPU vectorize:", timer()-start)
    if (n==10): print(b)

    print('-------------vs-------------')

    if (n==10): print(c)
    start = timer()
    c = addByGPU_jit(c)
    print("with GPU jit:", timer()-start)
    if (n==10): print(c)