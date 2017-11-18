import numpy as np
import h5py as h5
import pandas as pd

filename = "data/Xvalues.dat"

raw_data = pd.read_csv(filename,sep=',')

raw_data = np.asmatrix(raw_data)


n1,d1 = raw_data.shape

print(raw_data.shape)
# r,"c = x_data.shape

column_names = ["id","member_id","loan_amnt","funded_amnt","funded_amnt_inv","installment","emp_length",
				"home_ownership","annual_inc","verification_status","purpose","addr_state","dti","delinq_2yrs",
				"inq_last_6mths","mths_since_last_delinq","mths_since_last_record","open_acc","pub_rec","revol_bal",
				"revol_util","total_acc","initial_list_status","application_type","annual_inc_joint","dti_joint",
				"verification_status_joint","acc_now_delinq","tot_coll_amt","tot_cur_bal","open_acc_6m","open_il_6m",
				"open_il_12m","open_il_24m","mths_since_rcnt_il","total_bal_il","il_util","open_rv_12m","open_rv_24m",
				"max_bal_bc","all_util","total_rev_hi_lim","inq_fi","total_cu_tl","inq_last_12m"]

# Go through each column and see how many NaNs are in each
for j in range(d1):
	current_column = raw_data[:,j]
	nans = [i for i, x in enumerate(current_column) if x!=x]
	print("{},{}".format(column_names[j],len(nans)))	
