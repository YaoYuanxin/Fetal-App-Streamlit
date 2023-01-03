import sqlite3
from sqlalchemy import create_engine
import pandas as pd

# --- SQLITE3 DATABASE SETUP

connection = sqlite3.connect("fetal_app_database.db")
cursor = connection.cursor()

# Table for Non-Sequential Input
cursor.execute('CREATE TABLE IF NOT EXISTS\
                             non_sequential_input (wt_before_preg DOUBLE, height DOUBLE, NoPrevPreg NUMBER,\
                                                    hpb NUMBER, cardiac NUMBER, baseline_diabetes NUMBER,\
                                                    renal NUMBER, reg_smoke)')


# Table for Sequential Input

cursor.execute('CREATE TABLE IF NOT EXISTS\
                            sequential_input (gadays NUMBER, efw DOUBLE)')

connection.commit()



# --- IMPORT ORIGINAL DATA

covariates_5_record = pd.read_csv("Original Data/covariates_5_record.csv",index_col=0)
no_covariate_no_na_clean = pd.read_csv("Original Data/no_covariate_no_na_clean.csv",index_col=0)

covariates_5_record.set_index("id",inplace=True)
no_covariate_no_na_clean.set_index("id",inplace=True)


# --- PUT THE ORIGINAL DATA INTO THE DATABASES

covariates_5_record.to_sql('non_sequential_input', connection, if_exists='replace', index = True)

no_covariate_no_na_clean.to_sql('sequential_input', connection, if_exists='replace', index = True)


