"""
Handles USER INPUT and store USER INPUT into database
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import streamlit as st
from streamlit_option_menu import option_menu 
import pandas as pd
import numpy as np
from datetime import datetime    # use datetime to generate user_input id
import deta_database as db       # local import of database
from data_preprocess_utilities import *
import sqlite3

# -------------- SETTINGS --------------
page_title = "Fetal BirthWeight Prediction & LGA and Macrosomia Diagnosis"
page_icon = ":chart_with_upwards_trend:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "centered"
input_id = datetime.today().strftime("%b-%d-%Y") + "__" + datetime.now().strftime("%H-%M-%S")

mother_bios = ["Mother's Weight in **kg** Before Pregnancy",
                "Mother's Height in **cm**",
                "Number of Previous Pregnancies"]



# -------PAGE SETTINGS
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)

# --- DATABASE ENGINE FOR INPUT
# --- SQLITE3 DATABASE SETUP
# DataBase for Non-Sequential Input
connection = sqlite3.connect("app_database.db")
cursor = connection.cursor()
cursor.execute('CREATE TABLE IF NOT EXISTS\
                             non_sequential_input (wt_before_preg DOUBLE, height DOUBLE, NoPrevPreg NUMBER,\
                                                    hpb NUMBER, cardiac NUMBER, baseline_diabetes NUMBER,\
                                                    renal NUMBER, reg_smoke)')


# DataBase for Sequential Input
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




# --- MOTHER'S INFORMATION INPUT FORM
st.header(f"Enter Mother's Information and Health History:")
with st.form("Mother's Information", clear_on_submit=True):
    def non_sequential_input():
        with st.expander("_Mother's Biographical Information_"):
            wt_before_preg = st.number_input(f"Mother's Weight in **kg** Before Pregnancy \
                                            (Default value is the database average at 59.18kg)",
                                            min_value = 37.00,max_value=112.00, value=59.18)

            height = st.number_input(f"Mother's Height in **cm** \
                                    (Default value is the database average at 165.95cm )", 
                                    min_value = 147.00,max_value = 186.00, value = 165.95)

            NoPrevPreg = st.selectbox("Number of Previous Pregnancies", ("No previous pregancy.",
                                                                        "1 previous pregnancy.",
                                                                        "2 or more previous pregnancies."))
        
        with st.expander("_Mother's Health History_"):
            hpb = st.selectbox('Does the mother have **High Blood Pressure**?',('Yes','No'))
            cardiac = st.selectbox('Does the mother have **Cardiac Diseases**?',('Yes','No'))
            baseline_diabetes = st.selectbox('Does the mother have **Diabetes**?',('Yes','No'))
            renal = st.selectbox('Does the mother have **Renal Disorder**?',('Yes','No'))
            reg_smoke = st.selectbox('Is the mother a **Regular Smoker**?',('Yes','No'))



        non_sequential_input_data = {'wt_before_preg':wt_before_preg,
                    'height':height,
                    'NoPrevPreg':NoPrevPreg,
                    'hpb': hpb,
                    'cardiac': cardiac,
                    'baseline_diabetes': baseline_diabetes,
                    'renal': renal,
                    'reg_smoke': reg_smoke
    }

        non_sequential_input_df = pd.DataFrame(non_sequential_input_data, index=[0])
        non_sequential_input_df = non_sequential_input_df.replace({"Yes":1, "No":0})
        non_sequential_input_df.insert(0,column="id", value=input_id)
        non_sequential_input_df.set_index("id", inplace = True)
        non_sequential_input_df = non_sequential_input_df.replace({"No previous pregancy.":1, "1 previous pregnancy.":1, 
                                        "2 or more previous pregnancies.":2})
        return non_sequential_input_df

    input_df_mom = non_sequential_input()


    submitted_non_sequential = st.form_submit_button("**Save Mother's Information**")

    if submitted_non_sequential:
        # DISPLAY and STORE Non-Sequential Input
        st.subheader("***Mother's Information***")
        st.write("1 indicates 'yes' and 0 indicates 'no'.")
        st.dataframe(input_df_mom)
        input_df_mom.to_sql('non_sequential_input', con=connection, if_exists='append',index=True)
        st.success("Mother's Information saved!")
        #input_df_mom.to_csv("output.csv")


# --- ULTRASOUND INFO INPUT FORM 
st.header("Enter Fetal Ultrasound Measurements:")
with st.form("Ultrasound Measurements",clear_on_submit=True):
    def sequential_input_17():
        with st.expander(f"Measurements at **~17th Week**"):
            gadays = st.number_input(f"**Exact Gestational Age Days**",
                                        min_value = 84, max_value = 157, value = 122)

            bpd_mm = st.number_input(f"**Bipateral Diameter(BPD)** in milimeters",
                                        min_value = 20.00, max_value = 62.00, value = 39.86)

            mad_mm = st.number_input("**Middle Abdominal Diameter(MAD)** in milimeters",
                                         min_value = 20.00, max_value = 64.00, value = 37.91)

            fl_mm = st.number_input("**Femur Length(FL)** in milimeters",
                                        min_value = 0.00, max_value = 47.00, value = 24.57)

        sequential_input_data_17 = {
            "gadays":gadays,
            "bpd_mm":bpd_mm,
            "mad_mm":mad_mm,
            "fl_mm":fl_mm
        }
        sequential_input_df_17 = pd.DataFrame(sequential_input_data_17, index=[0])
        return sequential_input_df_17

    sequential_input_17_df = sequential_input_17()

    #Information at the 25th Week
    def sequential_input_25():
        with st.expander(f"Measurements at **~25th Week**"):
            gadays = st.number_input("**Exact Gestational Age Days**",
                                        min_value = 158, max_value = 201, value = 175)

            bpd_mm = st.number_input("**Bipateral Diameter(BPD)** in milimeters",
                                        min_value = 47.00, max_value = 96.00, value = 64.51)

            mad_mm = st.number_input("**Middle Abdominal Diameter(MAD)** in milimeters",
                                          min_value = 50.00,max_value = 109.00, value = 64.84)

            fl_mm = st.number_input("**Femur Length(FL)** in milimeters",
                                        min_value = 36.00, max_value = 71.00, value = 46.64)

        sequential_input_data_25 = {
            "gadays":gadays,
            "bpd_mm":bpd_mm,
            "mad_mm":mad_mm,
            "fl_mm":fl_mm
        }
        sequential_input_df_25 = pd.DataFrame(sequential_input_data_25, index=[0])
        return sequential_input_df_25

    sequential_input_25_df = sequential_input_25()

    #Information at the 33th Week
    def sequential_input_33():
        with st.expander(f"Measurements at **~33rd Week**"):
            gadays = st.number_input("**Exact Gestational Age Days**",
                                        min_value = 202, max_value = 246, value = 230)

            bpd_mm = st.number_input("**Bipateral Diameter(BPD)** in milimeters",
                                        min_value = 66.00, max_value = 98.00, value = 85.95)

            mad_mm = st.number_input("**Middle Abdominal Diameter(MAD)** in milimeters",
                                        min_value = 66.00, max_value = 111.00, value = 92.40)

            fl_mm = st.number_input("**Femur Length(FL)** in milimeters",
                                        min_value = 47.00, max_value = 74.00, value = 64.14)

        sequential_input_data_33 = {
            "gadays":gadays,
            "bpd_mm":bpd_mm,
            "mad_mm":mad_mm,
            "fl_mm":fl_mm
        }
        sequential_input_df_33 = pd.DataFrame(sequential_input_data_33, index=[0])
        return sequential_input_df_33

    sequential_input_33_df = sequential_input_33()

    #Information at the 37th Week
    def sequential_input_37():
        with st.expander(f"Measurements at **~37th Week**"):
            gadays = st.number_input("**Exact Gestational Age Days**",
                                        min_value = 247, max_value = 276, value = 259)

            bpd_mm = st.number_input("**Bipateral Diameter(BPD)** in milimeters",
                                          min_value = 78.00, max_value = 104.00, value = 92.97)

            mad_mm = st.number_input("**Middle Abdominal Diameter(MAD)** in milimeters",
                                        min_value = 85.00, max_value = 124.00, value = 104.97)

            fl_mm = st.number_input("**Femur Length(FL)** in milimeters",
                                        min_value = 52.00, max_value = 82.00, value = 71.51)

        sequential_input_data_37 = {
            "gadays":gadays,
            "bpd_mm":bpd_mm,
            "mad_mm":mad_mm,
            "fl_mm":fl_mm
        }
        sequential_input_df_37 = pd.DataFrame(sequential_input_data_37, index=[0])
        return sequential_input_df_37

    sequential_input_37_df = sequential_input_37()



    sequential_input_all = pd.concat([sequential_input_17_df,sequential_input_25_df,
                                    sequential_input_33_df,sequential_input_37_df],axis = 0)

    sequential_input_all['efw'] = efw19(sequential_input_all.bpd_mm,
                                        sequential_input_all.mad_mm,sequential_input_all.fl_mm)
    
    # gaweeks = ["17th Week", "25th Week", "33rd Week", "37th Week"]
    # sequential_input_all.insert(0, column = 'Gestational Week', value = gaweeks)
    # sequential_input_all.set_index("Gestational Week", inplace = True)
    
    gaweeks = [input_id+ str(17), input_id + str(25), input_id + str(33), input_id + str(37)]
    sequential_input_all.insert(0, column = 'Gestational Week', value = gaweeks)
    sequential_input_all.set_index("Gestational Week", inplace=True)    
    sequential_input_all = sequential_input_all[["gadays","efw"]]
    sequential_input_all.insert(0,column="id", value=input_id)
    sequential_input_all.set_index("id", inplace = True)
    submitted_sequential = st.form_submit_button("**Save Ultrasound Information**")

    if submitted_sequential:
        st.subheader("***Ultrasound Measurement Information***")
        st.write("The fetus' ultrasound measurements over 4 gestational age time steps. hmmm")
        st.dataframe(sequential_input_all)
        sequential_input_all.to_sql('sequential_input', con=connection, if_exists='append',index=True)
        st.success("Ultrasound Information saved!")

