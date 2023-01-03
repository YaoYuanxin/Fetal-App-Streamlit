import sqlite3
import pandas as pd

# --- SQLITE3 DATABASE SETUP
# DataBase for Non-Sequential Input
connection_non_sequential = sqlite3.connect("non_sequential.db")
cursor_non_sequential = connection_non_sequential.cursor()
cursor_non_sequential.execute('CREATE TABLE IF NOT EXISTS\
                             non_sequential_input (wt_before_preg DOUBLE, height DOUBLE, NoPrevPreg NUMBER,\
                                                    hpb NUMBER, cardiac NUMBER, baseline_diabetes NUMBER,\
                                                    renal NUMBER, reg_smoke)')

connection_non_sequential.commit()



# --- IMPORT ORIGINAL DATA

covariates_5_record = pd.read_csv("Original Data/covariates_5_record.csv",index_col=0)
no_covariate_no_na_clean = pd.read_csv("Original Data/no_covariate_no_na_clean.csv",index_col=0)

covariates_5_record.set_index("id",inplace=True)
no_covariate_no_na_clean.set_index("id",inplace=True)


# --- PUT THE ORIGINAL DATA INTO THE DATABASES

covariates_5_record.to_sql('non_sequential_input', connection_non_sequential, if_exists='replace', index = True)


# DataBase for Sequential Input
connection_sequential = sqlite3.connect("sequential.db")
cursor_sequential = connection_sequential.cursor()
cursor_sequential.execute('CREATE TABLE IF NOT EXISTS\
                            sequential_input (gadays NUMBER, efw DOUBLE)')


connection_sequential.commit()

no_covariate_no_na_clean.to_sql('sequential_input', connection_sequential, if_exists='replace', index = True)