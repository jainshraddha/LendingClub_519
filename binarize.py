import numpy as np
import h5py as h5
import pandas as pd

filename = "data/loan.csv"

raw_data = pd.read_csv(filename, sep=',')

raw_data = np.asmatrix(raw_data)
n1,d1 = raw_data.shape

# Binarized data column indexes
column_indexes = [12,14,23,35,52,55]

# Stores all the new columns for binarized features
new_columns = None


# Examine each column that needs to be binarized
for column_index in column_indexes:
	current_column = raw_data[:,column_indexes]

	# Get the unique values in the column
	unique_values = np.unique(current_column)
	num_vals =unique_values[0]

	# Create a new column for every unique value
	new_matrix = np.multiply(np.ones([n1,num_vals]),-1)
	value_index = 0
	for value in unique_values:
		# Get indices of rows with values matching the current unique value
		value_matches = [i for i, x in enumerate(current_column) if x == value]
		# Set the rows with matches for this value to 1 for the column representing this feature
		new_matrix[value_matches,value_index]=1
		value_index = value_index+1

	# Append the new columns to the overall list of columns
	if new_columns:
		new_columns = np.append(new_columns,new_matrix,axis=1)
	else:
		new_columns = new_matrix


# Remove the original non-binary columns from the original data
raw_data = np.delete(raw_data, column_indexes, 1)

# Add the new binarized columns to the data
binarized_data = np.append(raw_data,new_columns,1)




