import numpy as np
import h5py as h5
import pandas as pd

filename = "data/loan.csv"

raw_data = pd.read_csv(filename, sep=',')

raw_data = np.asmatrix(raw_data)
n1,d1 = raw_data.shape

column_indexes = [12,14,23,35,52,55]


for column_index in column_indexes:
	current_column = raw_data[:,column_indexes]
	unique_values = np.unique(current_column)
	n2,d2 =unique_values

	np.ones([])

	for value in unique_values:
		value_matches = [i for i, x in enumerate(current_column) if x == value]



