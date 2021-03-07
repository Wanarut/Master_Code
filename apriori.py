from apyori import apriori
import pandas as pd
from timeit import default_timer as timer

minimum_support = 0.004
minimum_confidence = 0.5
minimum_lift = 0
minimum_length = 2

# data = pd.read_csv('dataset/chess.dat', header=None, sep=" ")
data = pd.read_csv('dataset/store_data.csv', header=None, sep=",")
print(data.head())

# pre-processing data
records = []
for i in range(data.shape[0]):
    record = []
    for j in range(data.shape[1]):
        val = data.values[i, j]
        if pd.notnull(val):
            record.append(str(val))
    records.append(record)
print(records[0])

print('Start apriori algorithm')
start = timer()
association_rules = apriori(
    records, min_support=minimum_support, min_confidence=minimum_confidence, min_lift=minimum_lift, min_length=minimum_length)
association_rules = list(association_rules)
used_time = timer()-start

for item in association_rules:
    print("Rule:", list(item[2][0][0]), "->", list(item[2][0][1]))

#     # first index of the inner list
#     # Contains base item and add item
#     pair = item[0]
#     items = [x for x in pair]
#     print("Rule: " + items[0] + " -> " + items[1])

    # second index of the inner list
    print("Support: " + str(item[1]))

    # third index of the list located at 0th
    # of the third index of the inner list
    print("Confidence: " + str(item[2][0][2]))
    print("Lift: " + str(item[2][0][3]))
    print("=====================================")

print('\nminimum support:', minimum_support)
print('minimum confidence:', minimum_confidence)
print('minimum lift:', minimum_lift)
print('minimum length:', minimum_length)
print('There are', len(records), len(records[0]), 'transections')
print('Found', len(association_rules), 'rules')
print('Use', pd.to_timedelta(used_time, unit='s'), 'second\n')