import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth
from timeit import default_timer as timer

minimum_support = 0.4
minimum_confidence = 0
minimum_lift = 0
minimum_length = 2

data = pd.read_csv('dataset/chess.dat', header=None, sep=" ")
# data = pd.read_csv('dataset/store_data.csv', header=None, sep=",")
print(data.head())

# pre-processing data
dataset = []
for i in range(data.shape[0]):
    record = []
    for j in range(data.shape[1]):
        val = data.values[i, j]
        if pd.notnull(val):
            record.append(str(val))
    dataset.append(record)
print(dataset[0])

te = TransactionEncoder()
te_ary = te.fit(dataset).transform(dataset)
df = pd.DataFrame(te_ary, columns=te.columns_)
print(df.head())

print('Start fpgrowth algorithm')
start = timer()
association_rules = fpgrowth(df, min_support=minimum_support, use_colnames=True)
used_time = timer()-start

association_rules.to_csv('fpgrowth_ mlxtend_output.csv', index=False, header=True)
print(association_rules)

print('\nminimum support:', minimum_support)
print('minimum confidence:', minimum_confidence)
print('minimum lift:', minimum_lift)
print('minimum length:', minimum_length)
print('There are', len(dataset), len(dataset[0]), 'transections')
print('Found', len(association_rules), 'rules')
print('Use', pd.to_timedelta(used_time, unit='s'), 'second\n')
