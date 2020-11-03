import pandas as pd
import numpy as np
import os
from invesscience.joanna_target import get_company_target
from invesscience.joanna_12 import comps_founded_before
from invesscience.joanna_21 import merge_company_level
from invesscience.felipe_1 import time_serie_investment_new
from invesscience.felipe_10 import n_female_founders



ipos = pd.read_csv(os.path.join('..',"raw_data","ipos.csv"))
acq = pd.read_csv(os.path.join('..',"raw_data","acquisitions.csv"))
rounds = pd.read_csv(os.path.join('..',"raw_data","funding-rounds.csv"))
companies = pd.read_csv(os.path.join('..',"raw_data","companies.csv"))
relationships = pd.read_csv(os.path.join('..',"raw_data","relationships.csv"))
founders = pd.read_csv(os.path.join('..',"raw_data","founders.csv"))
people = pd.read_csv(os.path.join('..',"raw_data","people.csv"))
degrees = pd.read_csv(os.path.join('..',"raw_data","degrees.csv")).drop(columns=["updated_at","created_at"])



#get company table with target
reference = "a"
companies=get_company_target(ipos, acq, rounds,companies,reference)

#feature selection
companies = companies[["id",'normalized_name','category_code', "founded_at",\
'description', "closed_at",'exit', 'exit_date',\
  'country_code', "state_code",\
  f"date_series_{reference}", f"participants_{reference}", f"raised_amount_usd_{reference}"]]

#filter series a before certain date
companies = companies[companies[f"date_series_{reference}"]<'2009']
print(companies.head())
print(companies.shape)


#get time between founding date and reference round
companies.founded_at = pd.to_datetime(companies.founded_at)
companies[f'timediff_founded_series_{reference}'] = (companies[f"date_series_{reference}"] - companies.founded_at)/np.timedelta64(12, 'M')
print(companies.head())
print(companies.shape)

# get number of rounds before reference round (if reference is str if not not usefull)
if type(reference) == str:
    companies = time_serie_investment_new(rounds, companies, reference)

print(companies.head())
print(companies.shape)

#get diplomas of founding team
companies = merge_company_level(people, degrees,companies,relationships)

print(companies.head())
print(companies.shape)

#get female ratio in founders
companies = n_female_founders(companies, founders, relationships)

print(companies.head())
print(companies.shape)

# get mean companies founded before and total by founders
companies = comps_founded_before(companies, relationships, founders)#.sort_values(by="female_ratio",ascending=False)

print(companies.head(10))
print(companies.shape)
print(companies.columns)
print(companies.info())


