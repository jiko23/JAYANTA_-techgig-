import csv
from data_model import *
import pandas as pd
from sklearn.model_selection import train_test_split

#########################################################################################################################################################################################
"""
	READING THE DATA THAT WAS BEEN GENERATED BY converter.py PROGRAM AND CREATING DATAFRAME.
	READING DATA FROM 'cat_levels.csv' AND CREATING DATAFRAME.
	
"""
with open(r'E:\prog\Train_data\updated_train_data_update1.csv', encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file)
    rows = list(csv_reader)

header = rows.pop(0) 

data = pd.DataFrame(rows[1:500],columns=header)
key_words = data['keyword_set']

###################################
with open(r'E:\prog\Train_data\cat_levels.csv', encoding="utf8") as csv_file:
    csv_reader = csv.reader(csv_file)
    rows1 = list(csv_reader)

header = rows1.pop(0) 

categories = pd.DataFrame(rows1,columns=header)
cat_1 = categories['level1_categories']
cat_2 = categories['level_2_categories']

###################################################################################################################################################################################
"""
	RESULT SECTION
"""

pred1 = model_1(key_words,cat_1) ## UNSUPERVISED CATEGORY PREDICTION FOR KEYWORDS BASED ON cat_1 
pred2 = model_1(key_words,cat_2) ## UNSUPERVISED CATEGORY PREDICTION FOR KEYWORDS BASED ON cat_2

data["catg_1"] = ""
data["catg_2"] = ""

for i in range(0,len(pred1)):
	data.at[i,'catg_1'] = pred1[i] ## INSERTING PREDICTED CATEGORY_1 INTO 'data' DATAFRAME INTO COLUMN 'catg_1'
	data.at[i,'catg_2'] = pred2[i] ## INSERTING PREDICTED CATEGORY_2 INTO 'data' DATAFRAME INTO COLUMN 'catg_2'


result = final_model(data) ## FINAL CATORY CLASSIFICATION BASED ON 'catg_1' and 'catg_2'
result.to_csv(r'E:\prog\Train_data\Result.csv')







