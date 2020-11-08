
import pandas as pd
import numpy as np
import os


def clean_cat(x):
    if pd.isnull(x):
        return np.nan
    elif x in categories:
        return x
    else:
        return "other_category"

def clean_date(x):
    if pd.isnull(x):
        return np.nan
    elif str(x) > "1995":
        return str(x)
    else:
        return "pre_1996"

def clean_country(x):
    if pd.isnull(x):
        return np.nan
    elif x in countries:
        return x
    else:
        return "other_country"

def clean_state(x):
    if pd.isnull(x):
        return np.nan
    elif x in states:
        return x
    else:
        return "other_state"

def clean_founder(x):
    if pd.isnull(x):
        return np.nan
    elif x <= 3:
        return int(x)
    else:
        return 4

def clean_female(x):
    if pd.isnull(x):
        return np.nan
    elif x == 0:
        return 0
    else:
        return 1

def clean_training_data(companies):
    '''need to add the missing data Felipe completed'''
    '''neeed to check if same trends for series 0'''
    #cleaning categories - remove some to put in "other"
    categories = ["web", "mobile", "biotech", "advertising", "enterprise", "cleantech", "hardware", "network_hosting", "semiconductor", "search",\
             "security", "public_relations", "finance", "consulting", "analytics", "social", "music"]
    companies.category_code = companies.category_code.map(clean_cat)

    #cleaning founded_at to year and becomes cat variable, pre_1996 and after
    companies["founded_at"] = companies.founded_at.dt.strftime('%Y')
    companies.founded_year.map(clean_date)

    #cleaning country to most frequent and put rest in other_country
    countries = ["FRA","USA", "GBR", "CAN", "ISR", "DEU", "CHN", "SWE", "IND"]
    companies.country_code = companies.country_code.map(clean_country)

    #cleaning state to most frequent and put rest in other_state
    states = ['CA', 'NY', "MA", "WA", "TX", "CO", "VA", "IL", "PA", "MD", "NC", "NJ", "FL"]
    companies.state_code = companies.state_code.map(clean_state)

    #cleaning founder to new cat 4+ (=4)
    companies.founder_count = companies.founder_count.map(clean_founder)

    #cleaning female_ratio to 0 or 1
    companies["female_ratio"] = companies["female_ratio"].map(clean_female)




    #drop useless features
    companies = companies.drop(["n_female_founders"], axis=1)






