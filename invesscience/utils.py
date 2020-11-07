
import numpy as numpy
import time
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import os
from invesscience.joanna_merge import get_training_data
import pandas as pd


def compute_precision(y_pred, y_true):


    return precision_score(y_true, y_pred)

def compute_recall(y_pred, y_true):

     return recall_score(y_true, y_pred)

def compute_f1(y_pred, y_true):

     return f1_score(y_true, y_pred)

def simple_time_tracker(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts))
        else:
            print(method.__name__, round(te - ts, 2))
        return result

    return timed

def clean_data(reference):

    ''' quit nan observations with more than X nan values
        quit outliers'''

    path = os.path.dirname(os.path.dirname(__file__))

    companies= get_training_data(reference)

    #Drop not important columns
    companies = companies.drop(columns= ['normalized_name','founded_at', 'description' ,'exit', 'exit_date', f'date_series_{reference}', 'closed_at']).set_index('id')

    #quit observation with more than 5 nan
    companies = companies[companies.isnull().sum(axis=1) < 5]

    #quiting outliers
    companies = companies[companies[f'raised_amount_usd_{reference}']<=450000000]

    #Standarizing STATES
    #henri df
    df = pd.read_csv(os.path.join(path,"raw_data","datanamed_completed.csv"), sep=';', header=1)
    df = df[df.country_code=='USA'][['id', 'state_code']]

    merge_1 = companies.merge(df, how ='left', on = 'id')
    dict2 = merge_1.state_code_x.reset_index(drop=True).to_dict()
    dict1 = merge_1.state_code_y.reset_index(drop=True).to_dict()
    for i in dict1:
        if type(merge_1.state_code_y.reset_index(drop=True).to_dict()[i]) == type(""):
            dict2[i] = dict1[i]
        else:
            dict2[i] = 'other'
    merge_1['state_code'] = dict2.values()
    merge_1 = merge_1.drop(columns = ['state_code_y', 'state_code_x'])

    #Completing Countries columns

    df = pd.read_csv(os.path.join(path,"raw_data","countries_filled.csv"), sep=';')[['id','country_code', 'state_code']] #Warning: Modify the path!!!!


    merge_2 = merge_1.merge(df, how ='left', on = 'id')

    dict1 = merge_2.state_code_y.reset_index(drop=True).to_dict()
    dict2 = merge_2.state_code_x.reset_index(drop=True).to_dict()
    dict3 = merge_2.country_code_y.reset_index(drop=True).to_dict()
    dict4 = merge_2.country_code_x.reset_index(drop=True).to_dict()

    for i in dict1:
        if type(merge_2.state_code_y.reset_index(drop=True).to_dict()[i]) == type(""):
            dict2[i] = dict1[i]

    for i in dict3:
        if type(merge_2.country_code_y.reset_index(drop=True).to_dict()[i]) == type(""):
            dict4[i] = dict3[i]

    merge_2['state_code'] = dict2.values()
    merge_2['country_code'] = dict4.values()
    merge_2 = merge_2.drop(columns = ['state_code_y', 'state_code_x', 'country_code_y', 'country_code_x'])

    # Completing the categories

    df =pd.read_csv(os.path.join(path,"raw_data","categories_filled.csv"), sep=';')[['id', 'category_code']]

    merge_3 = merge_2.merge(df, how ='left', on = 'id')

    dict1 = merge_3.category_code_y.reset_index(drop=True).to_dict()
    dict2 = merge_3.category_code_x.reset_index(drop=True).to_dict()

    for i in dict1:
        if type(merge_3.category_code_y.reset_index(drop=True).to_dict()[i]) == type(""):
            dict2[i] = dict1[i]

    merge_3['category_code'] = dict2.values()
    merge_3 = merge_3.drop(columns = ['category_code_x', 'category_code_y'])


    if reference == 0:
        df = pd.read_csv(os.path.join(path,"raw_data","reference0_filled.csv"), sep=';')
        merge_3['state_code'] = df.state_code
        merge_3['country_code'] = df.country_code
        merge_3['category_code'] = df.category_code
        return merge_3.set_index('id')

    return merge_3.set_index('id')


