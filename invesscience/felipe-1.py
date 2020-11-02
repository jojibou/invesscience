import pandas as pd
import numpy as np
import os



def diff_foundation_fundround():
    comps = pd.read_csv(os.path.join('..',"raw_data","companies.csv"))
    comps = comps.rename(columns={'id': 'object_id'})
    rounds = pd.read_csv(os.path.join('..',"raw_data","funding-rounds.csv"))
    comps = comps.rename(columns={'id': 'object_id'})
    '''Returns a merge table between companies and funding rounds. One column added: difference between the date of the companies fundations
    and each round of investment. It includes companies_id repeated for each round that is realated with the company'''
    comps_rounds = comps[["object_id","founded_at"]].merge(rounds, how = 'right', on = 'object_id')
    comps_rounds.founded_at = pd.to_datetime(comps_rounds.founded_at)
    comps_rounds.funded_at = pd.to_datetime(comps_rounds.funded_at)
    comps_rounds['time_founded_invest'] = (comps_rounds.funded_at - comps_rounds.founded_at)/np.timedelta64(12, 'M')
    return comps_rounds

def time_serie_investments(rounds, reference='a'):
    ''' Take a serie X as reference: values = (a, b, c, angel, any on funding_round_code). And create a new column 'data_serie_reference' which contains the min date of this serie for each company.
    And also create a new column 'time_serie_investment' with the difference between each round and the date of reference serie'''
    rounds = pd.read_csv(os.path.join('..',"raw_data","funding-rounds.csv"))
    rounds.funded_at = pd.to_datetime(rounds.funded_at)

    tmp1 = rounds[rounds.funding_round_code == reference].sort_values(by="funded_at").groupby("object_id",as_index=False).first()
    tmp2 = rounds[rounds.funding_round_type == f'series-{reference}'].sort_values(by="funded_at").groupby("object_id",as_index=False).first()
    tmp3 = pd.concat([tmp1,tmp2])[["object_id","funded_at"]].dropna().drop_duplicates('object_id').rename(columns={'funded_at':f"date_series_{reference}"})
    rounds = rounds.merge(tmp3, how="left", on="object_id")
    rounds[f'time_serie_{reference}_investment'] =  -(rounds[f'date_series_{reference}'] - rounds.funded_at)/np.timedelta64(12, 'M')
    return rounds
