from fpgrowth_py import fpgrowth
import pandas as pd
from timeit import default_timer as timer

minimum_support = 0.04
minimum_confidence = 0.5
minimum_lift = 0
minimum_length = 2

data = pd.read_csv('dataset/chess.dat', header=None, sep=" ")
# data = pd.read_csv('dataset/store_data.csv', header=None, sep=",")
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

print('Start fpgrowth algorithm')
start = timer()
freqItemSet, association_rules = fpgrowth(
    records, minSupRatio=minimum_support, minConf=minimum_confidence)
used_time = timer()-start

for item in association_rules:
    print("Rule:", item[0], "->", item[1])
    print("Support: " + str(item[2]))
    print("=====================================")
