import pandas as pd
import numpy as np
import os
from invesscience.joanna_target import get_company_target



ipos = pd.read_csv(os.path.join('..',"raw_data","ipos.csv"))
acq = pd.read_csv(os.path.join('..',"raw_data","acquisitions.csv"))
rounds = pd.read_csv(os.path.join('..',"raw_data","funding-rounds.csv"))
companies = pd.read_csv(os.path.join('..',"raw_data","companies.csv"))

#get company table with target
reference = "a"
companies=get_company_target(ipos, acq, rounds,companies,reference)

#feature selection
companies = companies[["id",'normalized_name','category_code', "founded_at",\
'description',"status", "closed_at",'exit', 'exit_date',\
  'country_code', 'country_code_with_US', "state_code",\
  f"date_series_{reference}", f"participants_{reference}", f"raised_amount_usd_{reference}"]]

#filter series a before certain date
companies = companies[companies[f"date_series_{reference}"]<'2009']

print(companies.head())
print(companies.shape)
