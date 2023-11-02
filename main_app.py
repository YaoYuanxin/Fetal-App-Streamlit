"""
Handles USER INPUT and store USER INPUT into database
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime    # use datetime to generate user_input id
from data_preprocess_utilities import *
import subprocess
import plotly_express as px
import sqlite3

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
from sklearn import metrics
from keras.models import load_model
import base64



# -------------- SETTINGS --------------
page_title = "Birthweight and Excessive Fetal Growth Prediction   "
page_icon = ":chart_with_upwards_trend:"  # emojis: https://www.webfx.com/tools/emoji-cheat-sheet/
layout = "centered"
input_id = datetime.today().strftime("%m%d%Y") + datetime.now().strftime("%H%M%S")

mother_bios = ["Mother's Weight in **kg** Before Pregnancy",
                "Mother's Height in **cm**",
                "Number of Previous Pregnancies"]



# -------PAGE SETTINGS
st.set_page_config(page_title=page_title, page_icon=page_icon, layout=layout)
st.title(page_title + " " + page_icon)

st.markdown("""

:book: **_🔵 [Our Objective]:_**

**:red[Macrosomia]** refers to an infant's weight exceeding **4000 grams** at birth, regardless of the gestational age. This condition is often conflated with **:red["large for gestational age" (LGA)]**, which denotes a birth weight in the **upper 10% for the specific gestational age**. Both Macrosomia and LGA pose increased health risks to the mother and child during childbirth.

Accurate prediction of these conditions are crucial objectives in obstetric research. Early and accurate identification enables timely intervention and personalized medical care during childbirth, effectively reducing the associated risks.

:bulb: **_Instructions for Utilizing Our Tool:_** 

- Based on advanced Artificial Neural Networks from our study <*insert paper link*>, our tool achieves **94.7% accuracy** in predicting **:red[LGA]**, with a **75% true positive rate**. For **:red[macrosomia]** predictions, it provides **84% accuracy**, and a **96.8% true positive rate**.

- Ensure the entry forms below accurately reflect the medical records, with particular attention to **entry units**.

- In case of **missing specific information**, our system substitutes the missing entries with the most representative value from the database.

- Record typical ultrasound measurements during **4 crucial periods of gestation**: the **17th, 25th, 33rd, and 37th** week of gestation. If additional data points are available, prioritize the most accurate four measurements within these periods. Incomplete records will be supplemented by the most representative value from the database, maintaining prediction accuracy.

""")








# --- USER INPUT FORM

with st.form("User Input (2 Forms)", clear_on_submit=False):
    st.header(f" :page_facing_up: Maternal Baseline Characteristics:")
    def non_sequential_input():
        with st.expander("**Basic Information**"):
            wt_before_preg = st.number_input(f"Mother's Weight in **kg** Before Pregnancy ",
                                            min_value = 37.00,max_value=112.00, value=59.18, step=1.0)
            

            height = st.number_input(f"Mother's Height in **cm** ", 
                                    min_value = 147.00,max_value = 186.00, value = 165.95, step=1.0)

            NoPrevPreg = st.selectbox("Number of Previous Pregnancies", ("No previous pregancy.",
                                                                        "1 previous pregnancy.",
                                                                        "2 or more previous pregnancies."))
        
        with st.expander("**Mother's Health History**"):
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
    # st.subheader("***Mother's Information***")
    # st.write("1 indicates 'yes' and 0 indicates 'no'.")
    # st.write(input_df_mom)

    # --- FETAL ULTRASOUND MEASUREMENTS
    st.header(f":bar_chart: Prenatal Ultrasound Measurements:")
    def sequential_input_17():
        with st.expander(f"Measurements at **~17th Week**"):
            gadays = st.number_input(f"**Exact Days of Gestation**",
                                        min_value = 84, max_value = 157, value = 122, step=1)

            bpd_mm = st.number_input(f"**Bi-parietal Diameter(BPD)** in milimeters",
                                        min_value = 20.00, max_value = 62.00, value = 39.86, step=1.0)

            mad_mm = st.number_input("**Mid Abdominal Diameter(MAD)** in milimeters",
                                         min_value = 20.00, max_value = 64.00, value = 37.91, step=1.0)

            fl_mm = st.number_input("**Femur Length(FL)** in milimeters",
                                        min_value = 0.00, max_value = 47.00, value = 24.57, step=1.0)

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
            gadays = st.number_input("**Exact Days of Gestation**",
                                        min_value = 158, max_value = 201, value = 175, step=1)

            bpd_mm = st.number_input("**Bi-parietal Diameter(BPD)** in milimeters",
                                        min_value = 47.00, max_value = 96.00, value = 64.51, step=1.0)

            mad_mm = st.number_input("**Mid Abdominal Diameter(MAD)** in milimeters",
                                          min_value = 50.00,max_value = 109.00, value = 64.84, step=1.0)

            fl_mm = st.number_input("**Femur Length(FL)** in milimeters",
                                        min_value = 36.00, max_value = 71.00, value = 46.64, step=1.0)

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
            gadays = st.number_input("**Exact Days of Gestation**",
                                        min_value = 202, max_value = 246, value = 230, step=1)

            bpd_mm = st.number_input("**Bi-parietal Diameter(BPD)** in milimeters",
                                        min_value = 66.00, max_value = 98.00, value = 85.95, step=1.0)

            mad_mm = st.number_input("**Mid Abdominal Diameter(MAD)** in milimeters",
                                        min_value = 66.00, max_value = 111.00, value = 92.40, step=1.0)

            fl_mm = st.number_input("**Femur Length(FL)** in milimeters",
                                        min_value = 47.00, max_value = 74.00, value = 64.14, step=1.0)

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
            gadays = st.number_input("**Exact Days of Gestation**",
                                        min_value = 247, max_value = 276, value = 259, step=1)

            bpd_mm = st.number_input("**Bi-parietal Diameter(BPD)** in milimeters",
                                          min_value = 78.00, max_value = 104.00, value = 92.97, step=1.0)

            mad_mm = st.number_input("**Mid Abdominal Diameter(MAD)** in milimeters",
                                        min_value = 85.00, max_value = 124.00, value = 104.97, step=1.0)

            fl_mm = st.number_input("**Femur Length(FL)** in milimeters",
                                        min_value = 52.00, max_value = 82.00, value = 71.51, step=1.0)

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



    # DISPLAY and STORE User Input
    # Convert the single row DataFrame to a Series for easier indexing
    input_series = input_df_mom.iloc[0]

    def format_info(series):
        markdown_text = "## Basic Information\n\n"
        markdown_text += f"Mother’s Weight in kg Before Pregnancy: {series['Weight_kg']:.2f}\n"
        markdown_text += f"Mother’s Height in cm: {series['Height_cm']:.2f}\n"
        markdown_text += f"Number of Previous Pregnancies: {'No previous pregnancy.' if series['Previous_Pregnancies'] == 0 else f'{series['Previous_Pregnancies']} previous pregnancies.'}\n\n"
        markdown_text += "## Mother's Health History\n\n"
        markdown_text += f"Does the mother have High Blood Pressure? {series['High_Blood_Pressure']}\n"
        markdown_text += f"Does the mother have Cardiac Diseases? {series['Cardiac_Diseases']}\n"
        markdown_text += f"Does the mother have Diabetes? {series['Diabetes']}\n"
        markdown_text += f"Does the mother have Renal Disorder? {series['Renal_Disorder']}\n"
        markdown_text += f"Is the mother a Regular Smoker? {series['Regular_Smoker']}"
        
        return markdown_text


    submitted = st.form_submit_button("**Confirm Entries and Generate Results**")
    if submitted:
        st.markdown("Input submitted. Below is the summary:")
        st.markdown(format_info(input_series))
        st.dataframe(input_df_mom)
        st.dataframe(sequential_input_all)
        with st.spinner("Predicting Birthweights and Conditions..."):

            # Open DataBases ON SUBMISSION

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



            
            input_df_mom.to_sql('non_sequential_input', con=connection_non_sequential, if_exists='append',index=True)
            sequential_input_all.to_sql('sequential_input', con=connection_sequential, if_exists='append',index=True)
            

            
            
            process1 = subprocess.Popen(['Rscript', "test_r.R"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            result1 = process1.communicate()
            
            # Fetch the augmented data
            cursor_sequential.execute('''  
                                SELECT * FROM 'augmented_quad_df'
                            ''')

            augmented_data = pd.DataFrame(cursor_sequential.fetchall(), columns=["id","gadays","gadays_2","efw"])
            augmented_data.set_index("id",inplace=True)


            # Display the Augmented Data
            # st.write("**Displaying the Augmented Data.**")

            # fig, ax = plt.subplots()
            # ax.scatter(augmented_data.gadays, augmented_data.efw, c = 'b', label = "Augmented", alpha = 1 )
            # ax.scatter(no_covariate_no_na_clean.gadays,no_covariate_no_na_clean.efw, c = 'yellowgreen', label = "Original",alpha = 0.7 )


            # plt.xlabel("Gestational Age in Days")
            # plt.ylabel("Estimated Fetal Weight")
            # plt.title("Augmented Data vs Original Data (Quadratic Model)")
            # plt.legend()
            # st.pyplot(fig)


            # Ready the Augmented Data for RNN/LSTM
            new_daily_time = np.linspace(84,301,301-84+1)
            augmented_data_ready = preprocess_for_RNN_new(augmented_data)
            augmented_data_ready.columns = new_daily_time
            # st.write(augmented_data_ready)
            # st.write("Preprocess successful")
            
            y_days = 53
            n_features = 1
            y = augmented_data_ready.iloc[:,-y_days:]
            X = augmented_data_ready.drop(columns = y.columns)

            
            scaler_1 = StandardScaler()
            X_scaled = scaler_1.fit_transform(X)
            y_scaled = scaler_1.fit_transform(y)

            X_scaled_pred = X_scaled[-1,:]
            y_scaled_pred = y_scaled[-1,:]

            X_scaled_pred = X_scaled_pred.reshape(X_scaled_pred.shape[0],n_features)
            X_scaled_pred = X_scaled_pred.reshape(1,X_scaled_pred.shape[0],1)

            covariates_5_record = pd.concat([covariates_5_record,input_df_mom],axis = 0)
            scalar_2 = StandardScaler()
            covariates_5_record.wt_before_preg = scalar_2.fit_transform(np.array(covariates_5_record.wt_before_preg).reshape(-1,1))
            covariates_5_record.height = scalar_2.fit_transform(np.array(covariates_5_record.height).reshape(-1,1))
            input_df_mom_standardized = covariates_5_record.iloc[-1]
            input_df_mom_standardized = pd.DataFrame(input_df_mom_standardized).transpose()


            # Import Pre-trained RNN/LSTM models 
            
            model_qua_rnn_std_25 = load_model("model_qua_rnn_std_25.h5")
            pred_y = model_qua_rnn_std_25.predict([X_scaled_pred,input_df_mom_standardized])
            true_pred = scaler_1.inverse_transform(pred_y)
            true_pred_df = pd.DataFrame(true_pred)
            true_pred_df.columns = y.columns

            df_90th_10th = pd.read_csv("df_90th_10th.csv",index_col=0)


            lga_true = is_lga(true_pred_df,df_90th_10th)
            macro_true = is_macro(true_pred_df)
            lga_true.loc[0] = lga_true.loc[0].map({True: "Yes", False: "No"})
            macro_true.loc[0] = macro_true.loc[0].map({True: "Yes", False: "No"})

            st.header("Projected Birthweights and LGA/Macrosomia Conditions")
            result = pd.concat([true_pred_df,lga_true,macro_true],axis = 0)
            result.insert(0, column = "Result", value = ["Predicted Birthweight", "LGA Prediction", "Macrosomia Prediction"])
            result.set_index("Result",inplace=True)
            # st.write(result)
            # st.dataframe(result.loc[["LGA Prediction", "Macrosomia Prediction"]].style.apply(display_color_df, axis = 1))


            ## Interactive Plot 
            lower_bound = float(result.columns[0])
            upper_bound = float(result.columns[-1])
            lga_limit_df = df_90th_10th.loc[((df_90th_10th["gadays"]>=lower_bound) & (df_90th_10th["gadays"]<=upper_bound))]
            # st.write(lga_limit_df)
            # lga_limit_df.to_csv("lga_limit_df.csv")
            result_vert = result.transpose()
            lga_limit_df = lga_limit_df.set_index("gadays")
            result_vert.index =result_vert.index.astype("float64")

            overall_result = pd.concat([result_vert,lga_limit_df],axis = 1)
            overall_result["Predicted Birthweight"] = overall_result["Predicted Birthweight"].astype("float64")
            overall_result["90th percentile BW"] = overall_result["90th percentile BW"].astype("float64")
            overall_result["Macrosomoia Weight"] = 4000
            overall_result.insert(0,column="Gestational Age Day", value = overall_result.index)
            overall_result['Predicted Prediction'] = ''
            overall_result.loc[(overall_result['Macrosomia Prediction'] == 'No') & (overall_result['LGA Prediction'] == 'No'),\
                                'Predicted Prediction'] = 'Healthy'
            overall_result.loc[(overall_result['Macrosomia Prediction'] == 'Yes') & (overall_result['LGA Prediction'] == 'Yes'),\
                                'Predicted Prediction'] = 'Both LGA and Macrosomia'
            overall_result.loc[(overall_result['Macrosomia Prediction'] == 'No') & (overall_result['LGA Prediction'] == 'Yes'),\
                                'Predicted Prediction'] = 'LGA'
            overall_result.loc[(overall_result['Macrosomia Prediction'] == 'Yes') & (overall_result['LGA Prediction'] == 'No'),\
                                'Predicted Prediction'] = 'Macrosomia'
            
            overall_result = overall_result.drop(columns='10th percentile BW')
            overall_result[["Predicted Birthweight","90th percentile BW"]] = overall_result[["Predicted Birthweight","90th percentile BW"]].round(2)                                        
            st.dataframe(overall_result)

            def get_csv_download_link(df, filename="data.csv"):
                csv = df.to_csv(index=False)
                b64 = base64.b64encode(csv.encode()).decode()  # Encode to Base64
                href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Download CSV</a>'
                return href

            

            fig = px.scatter(overall_result, x= "Gestational Age Day", y=overall_result["Predicted Birthweight"], \
                            color = "Predicted Prediction", symbol = "Predicted Prediction")

            fig.add_scatter(x= overall_result["Gestational Age Day"], y= overall_result["90th percentile BW"], \
                            mode = "lines", name  = "LGA Threshold")

            fig.add_scatter(x= overall_result["Gestational Age Day"], y=overall_result['Macrosomoia Weight'],\
                            mode = "lines", name  = "Macrosmoia Threshold")

            fig.update_layout(width=1500, height=800,template="simple_white")
            # Show plot 
            st.plotly_chart(fig, use_container_width=True)

                    # Drop the Augmented Data from the DataBase
            cursor_sequential.execute('''  
                        DROP TABLE 'augmented_quad_df'
                    ''')
            connection_non_sequential.close()
            connection_sequential.close()
            

        st.success("Predictions successfully generated! Displaying projected **Birthweights** and **Predicted LGA/Macrosomia Conditions**.")
        st.markdown(f"""
            {get_csv_download_link(overall_result, filename="Predictions.csv")}
            Prediction Report is ready to download. 📊
            """, unsafe_allow_html=True)

        st.write("Feel free to modify the entries and re-run the analysis to explore different predictive outcomes.")

        





    

