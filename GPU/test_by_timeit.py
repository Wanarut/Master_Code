# execution time
from timeit import timeit
print(timeit('7. ** i', setup='i = 5'))
print(timeit('pow(7., i)', setup='i = 5'))
print(timeit('math.pow(7, i)', setup='import math; i = 5'))