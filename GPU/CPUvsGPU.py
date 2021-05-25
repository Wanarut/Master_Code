# conda install numba & conda install cudatoolkit

from numba import vectorize, jit, cuda
import numpy as np
# to measure exec time
from timeit import default_timer as timer

# normal function to run on cpu
def addByCPU(a):
    for i in range(a.size):
        a[i]+= 1
    return a
    # return a+1

# function optimized to run on gpu 
@vectorize(['float64(float64)'], target ="cuda")
def addByGPU(a):
    return a+1

if __name__=="__main__":
    n = 10000000
    # n = 10
    a = np.ones(n, dtype = np.float64)
    b = np.ones(n, dtype = np.float64)

    if (n==10): print(a)
    start = timer()
    a = addByCPU(a)
    print("without GPU:", timer()-start)
    if (n==10): print(a)

    print('-------------vs-------------')

    if (n==10): print(b)
    start = timer()
    b = addByGPU(b)
    print("with GPU:", timer()-start)
    if (n==10): print(b)