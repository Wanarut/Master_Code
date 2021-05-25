# conda install numba & conda install cudatoolkit

import numpy as np
from timeit import default_timer as timer
from numba import vectorize

def powerByCPU(a, b):
    return a ** b

@vectorize(['float32(float32, float32)'], target='cuda')
def powerByGPU(a, b):
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
    c = powerByGPU(a, b)
    print("with GPU:", timer()-start)
    if (n==3): print(c)