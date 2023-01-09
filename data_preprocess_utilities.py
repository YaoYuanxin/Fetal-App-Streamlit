"""
This file includes all the preprocess functions for input data.
"""
import math
import numpy as np
import pandas as pd
def efw19(bpd_mm,mad_mm,fl_mm):
    """
    calculates EFW from Empirical Formula 19,
    @param bpd_mm: bipateral diameter in milimeters
    @param mad_mm: middle abdominal diameter in milimeters
    @parm fl_mm: femur length in milimeters
    """
    BPD=bpd_mm*0.1
    MAD=mad_mm*0.1
    FL=fl_mm*0.1
    AC=math.pi*MAD
    EFW=10**(1.335-0.0034*AC*FL+0.0316*BPD+0.0457*AC+0.1623*FL)
    return EFW


def is_lga(y,df_90th_10th):
    """convert float list into a list of booleans
     (LGA), which describes an infant with a 90th percentile or
     higher birthweight for gestational age
     @param y
     @df_90th_10th
     """
    lower_bound = float(y.columns[0])
    upper_bound = float(y.columns[-1])
    lga_limit = df_90th_10th.loc[((df_90th_10th["gadays"]>=lower_bound) & (df_90th_10th["gadays"]<=upper_bound)),"90th percentile BW"]
    return y > np.array(lga_limit).reshape(1,53)

def is_macro(y):
    """convert float list into a list of booleans
    “macrosomia”is used to describe the condition of 
    a fetus with a birth weight of more than 4000 g."""
    return y>4000

def preprocess_for_RNN_new(df):
  """
  preprocess the data to be ready to use for RNN
     @param df: the input dataframe
  """
  result = []
  print("Preprocessing...")
  for id in df.index.unique():
    result.append(np.array(df[df.index == id].reset_index()['efw']).reshape(-1,))
  print("Done.")
  result = pd.DataFrame(result)
  result = result.set_index(df.index.unique())
  return result


def display_color_df(value):
  """
  Make streamlit display different colors when the predicted
    fetal birthweight has different dignoses

  """
  if value == "Yes":
    color = "red"
  else:
    color = "blue"
  return f'background-color: {color}'
