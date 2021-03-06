import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from apyori import apriori
from timeit import default_timer as timer

minimum_support = 0.7
minimum_confidence = 0.5
minimum_lift = 3
minimum_length = 2

data = pd.read_csv('dataset/chess.dat', header=None, sep=" ")
print(data.head())

records = []
for i in range(data.shape[0]):
    records.append([str(data.values[i, j]) for j in range(data.shape[1])])

print('\nThere are', len(records), len(records[0]), 'transections')

start = timer()

association_rules = apriori(
    records, min_support=minimum_support, min_confidence=minimum_confidence, min_lift=minimum_lift, min_length=minimum_length)
association_rules = list(association_rules)

used_time = timer()-start
print('Found', len(association_rules), 'rules')
print('Use', pd.to_timedelta(used_time, unit='s'), 'second\n')

for item in association_rules:

    # first index of the inner list
    # Contains base item and add item
    pair = item[0]
    items = [x for x in pair]
    print("Rule: " + items[0] + " -> " + items[1])

    # second index of the inner list
    print("Support: " + str(item[1]))

    # third index of the list located at 0th
    # of the third index of the inner list

    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")
