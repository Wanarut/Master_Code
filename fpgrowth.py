import pandas as pd
from timeit import default_timer as timer
from fpgrowth_py import fpgrowth

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
dataset = []
lines = open(file_name[0], 'r')
for line in lines:
    line = line.strip()
    if not line:
        continue
    dataset.append(line.split(file_name[1]))
print(dataset[0])

start = timer()
print('Start fpgrowth algorithm')
rules = []
try:
    freqItemSet, rules = fpgrowth(dataset, minSupRatio=minimum_support, minConf=minimum_confidence)   
except:
    print('fpgrowth algorithm Fail')
used_time = timer()-start

print('\nminimum support:', minimum_support)
print('minimum confidence:', minimum_confidence)
print('minimum lift:', minimum_lift)
print('minimum length:', minimum_length)
print('There are', len(dataset), len(dataset[0]), 'transections')
print('Found', len(rules), 'rules')
print('Use', pd.to_timedelta(used_time, unit='s'), 'second\n')

# print(rules)
if len(rules) > 0 :
    rules = pd.DataFrame(rules, columns=('antecedents','consequents','confidence'))
    print(rules.head())
    rules.to_csv('output_fpgrowth.csv',index=False, header=True)