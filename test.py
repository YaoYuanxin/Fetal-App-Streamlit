import deta_database as db       # local import of database
from data_preprocess_utilities import *

import pandas as pd


fetched_all_non_sequential = db.fetch_all_non_sequential()
print(type(fetched_all_non_sequential))

pd.DataFrame(fetched_all_non_sequential)