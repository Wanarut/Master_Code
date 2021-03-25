from apyori import apriori
import pandas as pd
from timeit import default_timer as timer

# file_name = ['dataset/T10I4D100K.dat', ' ']
# minimum_support = 0.005

# file_name = ['dataset/chess.dat', ' ']
# minimum_support = 0.7
# There are 3196 37 transections
# Found 8111370 rules
# Use 0 days 08:41:24.381572500 second

file_name = ['dataset/store_data.csv', ',']
minimum_support = 0.004

minimum_confidence = 0.5
minimum_lift = 1
minimum_length = 2

# pre-processing data
# dataset = []
# for i in range(data.shape[0]):
#     record = []
#     for j in range(data.shape[1]):
#         val = data.values[i, j]
#         if pd.notnull(val):
#             record.append(str(val))
#     dataset.append(record)
# print(dataset[0])

# pre-processing data
dataset = []
lines = open(file_name[0], 'r')
for line in lines:
    line = line.strip()
    if not line:
        continue
    dataset.append(line.split(file_name[1]))
print(dataset[0])

print('Start apriori algorithm')
start = timer()
rules = apriori(dataset, min_support=minimum_support, min_confidence=minimum_confidence,
                min_lift=minimum_lift, min_length=minimum_length)
# rules = Reader(rules)
rules = pd.DataFrame(rules)
used_time = timer()-start

print('\nminimum support:', minimum_support)
print('minimum confidence:', minimum_confidence)
print('minimum lift:', minimum_lift)
print('minimum length:', minimum_length)
print('There are', len(dataset), len(dataset[0]), 'transections')
print('Found', len(rules), 'rules')
print('Use', pd.to_timedelta(used_time, unit='s'), 'second\n')

if len(rules) > 0:
    ordered_statistics = rules['ordered_statistics']

    # print(list(ordered_statistics))
    result = []
    for item in ordered_statistics:
        result.append([list(item[0][0]), list(item[0][1]), item[0][2], item[0][3]])
    ordered_statistics = pd.DataFrame(result, columns=('antecedents','consequents','confidence','lift'))
    ordered_statistics['support'] = rules['support']
    print(ordered_statistics.head())

    ordered_statistics.to_csv('output_apriori.csv',index=False, header=True)