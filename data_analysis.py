import numpy as np
import h5py as h5
import pandas as pd

filename = "data/loan.csv"

raw_data = pd.read_csv(filename, sep=',')

raw_data = np.asmatrix(raw_data)
n1,d1 = raw_data.shape


# Go through each column and see how many NaNs are in each
for i in range(d1):
	current_column = raw_data[:,i]
	nans = [i for i, x in enumerate(current_column) if x!=x]
	print("Column {}: {} NaNs".format(i,len(nans)))