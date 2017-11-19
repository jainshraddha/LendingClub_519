import numpy as np
import h5py as h5
import pandas as pd

filename = "data/Xvalues.csv"
# filename = "binarizedX.csv"

raw_data = pd.read_csv(filename, sep=',')

raw_data = np.asmatrix(raw_data)
n1,d1 = raw_data.shape

# Binarized data column indexes
column_indexes = [7,9,10,11,20,21]
removed_headers = ['home_ownership','verification_status','purpose','addr_state','initial_list_status','application_type']

headers = ['id','member_id','loan_amnt','funded_amnt','funded_amnt_inv','installment','emp_length','annual_inc',
	'dti','delinq_2yrs','inq_last_6mths','open_acc','pub_rec	revol_bal','revol_util','total_acc','acc_now_delinq']

added_headers = []

# Stores all the new columns for binarized features
new_columns = None

ind = 0
# Examine each column that needs to be binarized
for column_index in column_indexes:
	current_column = np.array(raw_data[:,column_index])

	# Get the unique values in the column
	unique_values = np.unique(current_column)
	# num_vals =unique_values[0]
	num_vals = len(unique_values)

	# Create a new column for every unique value
	new_matrix = np.multiply(np.ones([n1,num_vals]),-1)
	value_index = 0

	current_header = removed_headers[ind]

	for value in unique_values:
		# Get indices of rows with values matching the current unique value
		value_matches = [i for i, x in enumerate(current_column) if x == value]
		# Set the rows with matches for this value to 1 for the column representing this feature
		new_matrix[value_matches,value_index]=1
		added_headers.append(current_header+"_"+value)
		value_index = value_index+1


	# Append the new columns to the overall list of columns
	if new_columns is None:
		new_columns = new_matrix
	else:
		new_columns = np.append(new_columns,new_matrix,axis=1)
	print("Finished column {}".format(column_index))
	ind = ind + 1


# Remove the original non-binary columns from the original data
raw_data = np.delete(raw_data, column_indexes, 1)

# Add the new binarized columns to the data
binarized_data = np.append(raw_data,new_columns,1)

final_headers = headers+added_headers

f = open('binarizedXValues.dat', 'w')

for i in range(len(final_headers)):
	f.write(final_headers[i]+",")

f.write("\n")

n,d = binarized_data.shape
for i in range(0,n):
	for j in range(0,d):
		num = binarized_data[i,j]
		part = str(num)
		f.write(part+",")
	f.write("\n")
f.close()




