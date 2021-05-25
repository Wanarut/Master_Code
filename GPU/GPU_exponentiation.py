# conda install numba & conda install cudatoolkit

import numpy as np
from timeit import default_timer as timer
from numba import vectorize, jit

def powerByCPU(a, b):
    return a ** b

@vectorize(['float32(float32, float32)'], target='cuda')
def powerByGPU_vectorize(a, b):
    return a ** b

@jit(nopython=True)
def powerByGPU_jit(a, b):
    return a ** b

if __name__ == '__main__':
    n = 100000000
    n = 3
    a = b = np.array(np.random.sample(n), dtype=np.float32)
    c = np.zeros(n, dtype=np.float32)
    
    if (n==3): print(a);print(b);print()

    start = timer()
    c = powerByCPU(a, b)
    print("without GPU:", timer()-start)
    if (n==3): print(c)

    print('-------------vs-------------')
    
    start = timer()
    c = powerByGPU_vectorize(a, b)
    print("with GPU vectorize:", timer()-start)
    if (n==3): print(c)

    print('-------------vs-------------')
    
    start = timer()
    c = powerByGPU_jit(a, b)
    print("with GPU jit:", timer()-start)
    if (n==3): print(c)