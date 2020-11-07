import multiprocessing
import time
import warnings
import category_encoders as ce
import joblib
import mlflow
import pandas as pd
import os
from memoized_property import memoized_property
from mlflow.tracking import MlflowClient
from psutil import virtual_memory
from sklearn.compose import ColumnTransformer
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from termcolor import colored
from xgboost import XGBRegressor
from invesscience.utils import compute_f1, simple_time_tracker, clean_data, compute_precision
from invesscience.joanna_merge import get_training_data
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression



MLFLOW_URI = "https://mlflow.lewagon.co/"


class Trainer(object):
    ESTIMATOR = "LogisticRegression"
    EXPERIMENT_NAME = "Invesscience_batch_#463"
    IMPUTER = 'SimpleImputer'
    SCALER_AMOUNT = 'RobustScaler'
    SCALER_PROFESSIONALS = 'MinMaxScaler'
    SCALER_TIME = 'StandardScaler'
    SCALER_PARTICIPANTS = 'StandardScaler'

    def __init__(self, X, y, **kwargs):
        """
        FYI:
        __init__ is called every time you instatiate Trainer
        Consider kwargs as a dict containig all possible parameters given to your constructor
        Example:
            TT = Trainer(nrows=1000, estimator="Linear")
               ==> kwargs = {"nrows": 1000,
                            "estimator": "Linear"}
        :param X:
        :param y:
        :param kwargs:
        """
        self.pipeline = None
        self.kwargs = kwargs
        self.local = kwargs.get("local", False)  # if True training is done locally
        self.mlflow = kwargs.get("mlflow", False)
        self.reference = kwargs.get("reference", 'a')
        self.tag = kwargs.get("tag_description", "nada")
        self.experiment_name = kwargs.get("experiment_name", self.EXPERIMENT_NAME)  # cf doc above
        self.model_params = None  # for
        self.X_train = X
        self.y_train = y
        del X, y
        self.split = self.kwargs.get("split", True)  # cf doc above
        if self.split:
            self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(self.X_train, self.y_train,
                                                                                  test_size=0.3)
        self.nrows = self.X_train.shape[0]  # nb of rows to train on
        self.log_kwargs_params()
        self.log_machine_specs()

    def get_estimator(self):
        estimator = self.kwargs.get("estimator", self.ESTIMATOR)
        if estimator == "LogisticRegression":
            model = LogisticRegression()
        elif estimator == "SVC":
            model = SVC()
        elif estimator == "KNeighborsClassifier":
            model = KNeighborsClassifier()
        elif estimator == "DecisionTree":
            model = DecisionTreeClassifier()

        elif estimator == "RandomForestClassifier":
            model = RandomForestClassifier()
            self.model_params = {  # 'n_estimators': [int(x) for x in np.linspace(start = 50, stop = 200, num = 10)],
                'max_features': ['auto']}
            # 'max_depth' : [int(x) for x in np.linspace(10, 110, num = 11)]}


        #elif estimator == "xgboost":
            #model = XGBRegressor(objective='reg:squarederror', n_jobs=-1, max_depth=10, learning_rate=0.05,
                                 #gamma=3)

        #else:
            #model = Lasso()


        estimator_params = self.kwargs.get("estimator_params", {}) #Dictionary
        self.mlflow_log_param("estimator", estimator)
        model.set_params(**estimator_params)
        print(colored(model.__class__.__name__, "red"))
        return model

    def get_imputer(self):
        imputer = self.kwargs.get("imputer", self.IMPUTER)
        if imputer == "SimpleImputer":
            imputer_use = SimpleImputer()
        if imputer == "KNNImputer":
            imputer_use = KNNImputer()


        imputer_params = self.kwargs.get("imputer_params", {})
        self.mlflow_log_param("imputer", imputer)
        imputer_use.set_params(**imputer_params)
        print(colored(imputer_use.__class__.__name__, "blue"))

        return imputer_use



    def get_scaler_raised_amount(self):
        scaler_amount = self.kwargs.get("scaler_amount", self.SCALER_AMOUNT)
        if scaler_amount == "RobustScaler":
            scaler_use = RobustScaler()
        elif scaler_amount == "StandardScaler":
            scaler_use = StandardScaler()
        elif scaler_amount == 'MinMaxScaler':
            scaler_use = MinMaxScaler()

        scaler_amount_params = self.kwargs.get("scaler_amount_params", {})
        self.mlflow_log_param("scaler_amount", scaler_amount)
        scaler_use.set_params(**scaler_amount_params)
        print(colored(scaler_use.__class__.__name__, "blue"))

        return scaler_use


    def get_scaler_professionals(self):
        scaler_professionals = self.kwargs.get("scaler_professionals", self.SCALER_PROFESSIONALS)
        if scaler_professionals == "RobustScaler":
            scaler_use = RobustScaler()
        elif scaler_professionals == "StandardScaler":
            scaler_use = StandardScaler()
        elif scaler_professionals == 'MinMaxScaler':
            scaler_use = MinMaxScaler()

        scaler_professionals_params = self.kwargs.get("scaler_professionals_params", {})
        self.mlflow_log_param("scaler_professionals", scaler_professionals)
        scaler_use.set_params(**scaler_professionals_params)
        print(colored(scaler_use.__class__.__name__, "blue"))

        return scaler_use

    def get_scaler_time(self):
        scaler_time = self.kwargs.get("scaler_time", self.SCALER_TIME)
        if scaler_time == "RobustScaler":
            scaler_use = RobustScaler()
        elif scaler_time == "StandardScaler":
            scaler_use = StandardScaler()
        elif scaler_time == 'MinMaxScaler':
            scaler_use = MinMaxScaler()

        scaler_time_params = self.kwargs.get("scaler_time_params", {})
        self.mlflow_log_param("scaler_time", scaler_time)
        scaler_use.set_params(**scaler_time_params)
        print(colored(scaler_use.__class__.__name__, "blue"))

        return scaler_use


    def get_scaler_participant(self):
        scaler_participants = self.kwargs.get("scaler_participants", self.SCALER_PARTICIPANTS)
        if scaler_participants == "RobustScaler":
            scaler_use = RobustScaler()
        elif scaler_participants == "StandardScaler":
            scaler_use = StandardScaler()
        elif scaler_participants == 'MinMaxScaler':
            scaler_use = MinMaxScaler()

        scaler_participant_params = self.kwargs.get("scaler_participant_params", {})
        self.mlflow_log_param("scaler_participants", scaler_participants)
        scaler_use.set_params(**scaler_participant_params)
        print(colored(scaler_use.__class__.__name__, "blue"))

        return scaler_use


    def set_pipeline(self):

        #Column spliting
        if self.reference=='a':
            categorical_features = list(self.X_train.select_dtypes('object').columns)

            top_features_num = ['top_5', 'top_20','top_50']
            #boolean_features = ['MBA_bool', 'cs_bool', 'phd_bool' ,'top_5_bool', 'top_20_bool', 'top_50_bool']
            investment_amount_features = [f'raised_amount_usd_{self.reference}', f'raised_before_{self.reference}', f'rounds_before_{self.reference}' ]
            time_feature = [f'timediff_founded_series_{self.reference}']
            participant_feature = [f'participants_{self.reference}', f'participants_before_{self.reference}']
            professional_features = ['phd', 'MBA', 'cs','graduate', 'undergrad',
                                    'professional', 'degree_count','founder_count',
                                    'n_female_founders','female_ratio', 'mean_comp_founded_ever',
                                    'mean_comp_founded_before']

        elif self.reference==0:
            categorical_features = list(self.X_train.select_dtypes('object').columns)

            top_features_num = ['top_5', 'top_20','top_50']
            #boolean_features = ['MBA_bool', 'cs_bool', 'phd_bool' ,'top_5_bool', 'top_20_bool', 'top_50_bool']
            investment_amount_features = [f'raised_amount_usd_{self.reference}']
            time_feature = [f'timediff_founded_series_{self.reference}']
            participant_feature = [f'participants_{self.reference}']
            professional_features = ['phd', 'MBA', 'cs','graduate', 'undergrad',
                                    'professional', 'degree_count','founder_count',
                                    'n_female_founders','female_ratio', 'mean_comp_founded_ever',
                                    'mean_comp_founded_before']


        #Defining imputers

        notdegrees_imputer = self.get_imputer()
        raised_amount_scaler = self.get_scaler_raised_amount()
        profesionals_scaler = self.get_scaler_professionals()
        timediff_scaler = self.get_scaler_time()
        participant_scaler = self.get_scaler_participant()


        #pipes for each feature

        pipe_amounts = Pipeline([('raised_amount_imputer', notdegrees_imputer),
                                 ('raised_amount_scaler', raised_amount_scaler)])

        pipe_categorical = Pipeline([('ohe', OneHotEncoder(handle_unknown='ignore'))])

        pipe_professionals = Pipeline([('profesionals_scaler', profesionals_scaler)])

        pipe_time = Pipeline([('time_imput', notdegrees_imputer),
                              ('timediff_scaler', timediff_scaler)])

        pipe_participants =  Pipeline([('participant_imputer', notdegrees_imputer ),
                                      ('participant_scaler', participant_scaler)])

        #process

        feateng_blocks = [ ('participant_scaler', pipe_participants, participant_feature),
                           ('investment_scaler', pipe_amounts, investment_amount_features),# cambiar en caso de 0
                           ('timediff_scaler', pipe_time, time_feature),
                           ('profesionals_scaler', pipe_professionals, professional_features),
                           ('top_scale', MinMaxScaler(), top_features_num), #just to stablish order of output columns
                           #('bolean_var',  MinMaxScaler(), boolean_features), #just to stablish order of output columns
                           ('cat_pipe', pipe_categorical, categorical_features)]



        #Columntransformer keeping order
        preprocessor = ColumnTransformer(feateng_blocks)

        #final_pipeline
        self.pipeline = Pipeline(steps = [('preprocessing', preprocessor),
                            ('rgs', self.get_estimator())] )


    @simple_time_tracker
    def train(self):
        tic = time.time()
        self.set_pipeline()
        self.pipeline.fit(self.X_train, self.y_train)
        # mlflow logs
        self.mlflow_log_metric("train_time", int(time.time() - tic))
        self.set_tag('tag_instance', self.tag)

    def evaluate(self):
        f1_train = self.compute_f1(self.X_train, self.y_train)
        precision_train = self.compute_precision(self.X_train, self.y_train)

        self.mlflow_log_metric("f1score_train", f1_train)
        self.mlflow_log_metric("precision_train", precision_train)

        if self.split:
            f1_val = self.compute_f1(self.X_val, self.y_val, show=True)
            precision_val = self.compute_precision(self.X_val, self.y_val, show=True)
            self.mlflow_log_metric("f1score_val", f1_val)
            self.mlflow_log_metric("precision_val", precision_val)
            print(colored("f1 train: {} || f1 val: {}".format(f1_train, f1_val), "yellow"))
            print(colored("precision train: {} || precision val: {}".format(precision_train, precision_val), "yellow"))
        else:
            print(colored("f1 train: {}".format(f1_train), "blue"))
            print(colored("precision train: {}".format(precision_train), "blue"))

    def compute_f1(self, X_test, y_test, show=False):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        if show:
            res = pd.DataFrame(y_test)
            res["pred"] = y_pred
            print(colored(res.sample(self.y_val.shape[0]), "blue")) #Aumentar tamaño de muestra de validacion
        f1 = compute_f1(y_pred, y_test)
        return round(f1, 3)


    def compute_precision(self, X_test, y_test, show=False):
        if self.pipeline is None:
            raise ("Cannot evaluate an empty pipeline")
        y_pred = self.pipeline.predict(X_test)
        if show:
            res = pd.DataFrame(y_test)
            res["pred"] = y_pred
            print(colored(res.sample(self.y_val.shape[0]), "blue")) #Aumentar tamaño de muestra de validacion
        precision = compute_precision(y_pred, y_test)
        return round(precision, 3)


    def save_model(self):
        """Save the model into a .joblib format"""
        joblib.dump(self.pipeline, 'model.joblib')
        print(colored("model.joblib saved locally", "green"))

    ### MLFlow methods
    @memoized_property
    def mlflow_client(self):
        mlflow.set_tracking_uri(MLFLOW_URI)
        return MlflowClient()

    @memoized_property
    def mlflow_experiment_id(self):
        try:
            return self.mlflow_client.create_experiment(self.experiment_name)
        except BaseException:
            return self.mlflow_client.get_experiment_by_name(self.experiment_name).experiment_id

    @memoized_property
    def mlflow_run(self):
        return self.mlflow_client.create_run(self.mlflow_experiment_id)

    def mlflow_log_param(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_param(self.mlflow_run.info.run_id, key, value)

    def mlflow_log_metric(self, key, value):
        if self.mlflow:
            self.mlflow_client.log_metric(self.mlflow_run.info.run_id, key, value)

    def log_estimator_params(self):
        reg = self.get_estimator()
        self.mlflow_log_param('estimator_name', reg.__class__.__name__)
        params = reg.get_params()
        for k, v in params.items():
            self.mlflow_log_param(k, v)

    def set_tag(self, key, value):
        if self.mlflow:
            self.mlflow_client.set_tag(self.mlflow_run.info.run_id, key, value)

    def log_kwargs_params(self):
        if self.mlflow:
            for k, v in self.kwargs.items():
                self.mlflow_log_param(k, v)

    def log_machine_specs(self):
        cpus = multiprocessing.cpu_count()
        mem = virtual_memory()
        ram = int(mem.total / 1000000000)
        self.mlflow_log_param("ram", ram)
        self.mlflow_log_param("cpus", cpus)





if __name__ == "__main__":
    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Get and clean data
    experiment = "Invesscience_batch_#463"


    #Change the reference HERE !!!


    reference = 'a'
    columnas = [f'participants_{reference}',f'participants_before_{reference}',f'raised_amount_usd_{reference}', f'raised_before_{reference}', f'rounds_before_{reference}',
                                                        f'timediff_founded_series_{reference}', 'phd', 'MBA', 'cs',
                                                        'graduate', 'undergrad', 'professional', 'degree_count',
                                                        'founder_count', 'n_female_founders','female_ratio',
                                                        'mean_comp_founded_ever', 'mean_comp_founded_before', 'top_5', 'top_20','top_50',
                                                        #'MBA_bool', 'cs_bool', 'phd_bool' ,'top_5_bool', 'top_20_bool', 'top_50_bool',
                                                        'state_code', 'country_code', 'category_code','target' ]



    # grid_search = GridSearchCV(
    #     final_pipe,
    #     param_grid={
    #         # Access any component of the pipeline, as far back as you want
    #         'preprocessing__agetransformer__imputer__strategy': ['mean', 'median'],
    #         'linear_regression__normalize': [True, False]},
    #     cv=5,
    #     scoring="r2")

    # grid_search.fit(X_train, y_train)




    for estimator_iter in ['LogisticRegression']:




        params = dict(tag='trying', reference =reference ,estimator = estimator_iter, estimator_params ={'class_weight': 'balanced'}, local=False, split=True,  mlflow = True,
            experiment_name=experiment,imputer= 'KNNImputer', imputer_params = {}, scaler_professionals= 'MinMaxScaler' , scaler_professionals_params = {},
         scaler_time= 'StandardScaler', scaler_time_params={}, scaler_amount='MinMaxScaler', scaler_amount_params={} , scaler_participants='RobustScaler',
         scaler_participant_params={} ) #agregar



        print("############   Loading Data   ############")

        df = clean_data(reference)
        df = df[columnas]

        y_train = df["target"]
        X_train = df.drop(columns =['target']) #Change when we have categorical var
        del df
        print("shape: {}".format(X_train.shape))
        print("size: {} Mb".format(X_train.memory_usage().sum() / 1e6))
        # Train and save model, locally and
        t = Trainer(X=X_train, y=y_train, **params)
        del X_train, y_train

        print(type(t.set_pipeline()))

        # print(colored("############  Training model   ############", "red"))
        # t.train()
        # print(colored("############  Evaluating model ############", "blue"))
        # t.evaluate()
        # print(colored("############   Saving model    ############", "green"))
        # t.save_model()
