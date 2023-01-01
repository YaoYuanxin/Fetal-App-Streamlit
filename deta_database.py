import os

from deta import Deta
from dotenv import load_dotenv  #pip install python-dotenv

# Load the environment
load_dotenv(".env")
DETA_KEY = os.getenv("DETA_KEY")

deta = Deta(DETA_KEY)


# DATABASE FOR MOM'S INFORMATION/ Non-Sequential Input
db_mom = deta.Base("non_sequential")

def store_non_sequential(non_sequential_input_df):
    return db_mom.put(non_sequential_input_df)

def fetch_all_non_sequential():
    feched_data = db_mom.fetch()
    return feched_data.items



# DATABASE FOR EFW/ Sequential Input

db_efw = deta.Base("sequential")

def store_sequential(sequential_input_df):
    return db_efw.put(sequential_input_df)

def fetch_all_non_sequential():
    feched_data = db_efw.fetch()
    return feched_data.items

def get_id(input_id):
    """
    If the id is not found in the DataBase, returns None
    """
    return db_mom(input_id)
