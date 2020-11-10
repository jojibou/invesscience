
import numpy as numpy
import time
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.metrics import f1_score
import os
from invesscience.joanna_merge import get_training_data
import pandas as pd
from invesscience.joanna_clean_data_felipe import clean_training_data


def compute_precision_cv(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division=1):


    return precision_score(y_true, y_pred, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division=1)

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


def get_data_filled(reference = 'a', target_to_drop ='exit' , year = '2014'):

    ''' just working for reference A for the moment '''
    path = os.path.dirname(os.path.dirname(__file__))

    #After feature selection
    if year =='2014':
        features_a=['id',
                    'category_code', 'country_code','state_code', 'founded_at','timediff_founded_series_a', 'time_diff_series_a_now', 'founder_count',
                    'rounds_before_a' , 'raised_amount_usd_a','participants_a', 'mean_comp_worked_before',
                    'graduate',  'MBA_bool', 'cs_bool', 'top_20_bool', 'female_ratio', 'exit', 'target']




    if year == '2009':
        features_a=['id',
                    'category_code', 'country_code','state_code', 'founded_at','timediff_founded_series_a',  'founder_count',
                    'rounds_before_a' , 'raised_amount_usd_a','participants_a', 'mean_comp_worked_before',
                    'graduate',  'MBA_bool', 'cs_bool', 'top_20_bool', 'female_ratio', 'exit', 'target']



    companies_total = get_training_data(reference=reference, cut=year)
    companies_total = clean_training_data(companies_total, reference=reference)

    #df = pd.read_csv(os.path.join(path, 'raw_data' , 'last_complete_a.csv'), sep=';')
    companies_total_filled_a = companies_total[features_a][companies_total[features_a].isnull().sum(axis = 1)<3].reset_index(drop=True)
    #companies_total_filled_a['country_code'] = df['country_code']
    #companies_total_filled_a['state_code'] = df['state_code']

    companies_total_filled_a = companies_total_filled_a[companies_total_filled_a['category_code'].notna()]


    return companies_total_filled_a.set_index('id').drop(columns = [target_to_drop])




