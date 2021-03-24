import pandas as pd
from timeit import default_timer as timer
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import apriori, fpgrowth, fpmax, association_rules

# file_name = ['dataset/T10I4D100K.dat', ' ']
# minimum_support = 0.005

# file_name = ['dataset/chess.dat', ' ']
# minimum_support = 0.7
# There are (3196, 75) transections
# Found 7060259 rules
# Use 0 days 00:01:26.117246400 second

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
# print(dataset[0])
te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df.head())

start = timer()
print('Start mlxtend algorithm')
rules = []
try:
    # freqItemSet = apriori(df, min_support=minimum_support, use_colnames=True)
    freqItemSet = fpgrowth(df, min_support=minimum_support, use_colnames=True)
    # freqItemSet = fpmax(df, min_support=minimum_support, use_colnames=True)
    rules = association_rules(freqItemSet, metric='confidence', min_threshold=minimum_confidence) 
except:
    print('mlxtend algorithm Fail')
used_time = timer()-start

print('\nminimum support:', minimum_support)
print('minimum confidence:', minimum_confidence)
print('minimum lift:', minimum_lift)
print('minimum length:', minimum_length)
print('There are', df.shape, 'transections')
print('Found', len(rules), 'rules')
print('Use', pd.to_timedelta(used_time, unit='s'), 'second\n')

if len(rules) > 0 :
    print(rules.head())
    rules.to_csv('mlxtend_output.csv',index=False, header=True)