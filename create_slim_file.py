import numpy as np
import h5py as h5
import pandas as pd

filename = "data/loan.csv"

raw_data = pd.read_csv(filename, sep=',')

raw_data = np.asmatrix(raw_data)

print("the raw data shape is: ", raw_data.shape, type(raw_data))

d = [5, 10, 18, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

slimData = np.delete(raw_data, d, 1)

print slimData[5]
print slimData[6]

print slimData[5, 13]
print slimData[6, 13]

#test_list = ["Dec-09", "Jan-07", "May-08"]
#for i in range(len(test_list)):
#    if "7" in test_list[i] or "8" in test_list[i] or "9" in test_list[i]:
#        print("found it!")


n, d = slimData.shape
print("slimData shape is", slimData.shape, type(slimData))

filename = "data/slim_data.dat"

slim_data_file = open(filename, "w")

del_rows = []
#13 is the column with issue dates, after 2 columns are deleted
for i in range(n):
    date = slimData[i, 13]
    if "7" in date or "8" in date or "9" in date:
#        del_rows.append(i)
        continue
    print slimData[i]
#    slim_data_file.write(slimData[i])
#    slim_data_file.write(','.join(map(repr, slimData[i])))
    slim_data_file.write('\n')
