import numpy as np
import re
import pandas as pd
from time import time
from scipy.stats import skew,randint

from sklearn.model_selection import train_test_split,cross_val_score,KFold,StratifiedKFold,ShuffleSplit,StratifiedShuffleSplit, GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import PolynomialFeatures,LabelEncoder,Imputer,RobustScaler, StandardScaler, MinMaxScaler,FunctionTransformer
from sklearn.ensemble import RandomForestRegressor,GradientBoostingRegressor
from sklearn.metrics import mean_squared_error,mean_squared_log_error,r2_score

def extract_and_drop_target_column(df_in, y_name, inplace=False):
    if(inplace):
        df = df_in
    else:
        df = df_in.copy()

    y = df[y_name].copy()
    df.drop([y_name], axis=1, inplace=True)
    return (df,y)



def process_date_column(df_in, colname, include_time=False, inplace=True,
                        date_format=None):
    if(inplace):
        df = df_in
    else:
        df = df_in.copy()

    date_column = 'Date'
    if(df[colname].dtype != 'datetime64[ns]'):
        if date_format is not None:
            df[colname] = pd.to_datetime(df[colname],format=date_format)
        else:
            df[colname] = pd.to_datetime(df[colname],infer_datetime_format=True)
    columns = ['Year', 'Month', 'Week','Day',
               'Dayofweek', 'Dayofyear',
               'Is_month_end','Is_month_start',
               'Is_quarter_end','Is_quarter_start',
               'Is_year_end','Is_year_start']
    if include_time:
        columns = columns + ['Hour', 'Minute', 'Second']
    for c in columns:
        df[date_column + '_' + c] = getattr(df[colname].dt,c.lower())
    df[date_column] = df[colname].astype(np.int64) // (10 ** 9)
    df.drop(colname,axis=1,inplace=True)

    return df

def simple_split(df,y,n):
    X_train =  df[:n].copy()
    X_test = df[n:].copy()
    y_train = y[:n].copy()
    y_test  = y[n:].copy()
    return X_train,X_test,y_train,y_test

def get_train_valid(df,y,test_size):

    test_size = df.shape[0] * (20/100)
    X_train,X_test,y_train,y_test = simple_split(df,y,(df.shape[0] - test_size))
    print(X_train.shape,X_test.shape,y_train.shape,y_test.shape)
    X_train,X_valid,y_train,y_valid = simple_split(X_train,y_train,
                                                   X_train.shape[0] - test_size)
    print(X_train.shape,X_valid.shape,y_train.shape,y_valid.shape)

    return

def print_mse(m,X_train, X_valid, y_train, y_valid):
    res = [mean_squared_error(y_train,m.predict(X_train)),
                mean_squared_error(y_valid,m.predict(X_valid)),
                m.score(X_train, y_train), m.score(X_valid, y_valid)]
    print('MSE Training set = {}, MSE Validation set = {}, score Training Set = {}, score on Validation Set = {}'.format(res[0],res[1],res[2], res[3]))
    if hasattr(m, 'oob_score_'):
          print('OOB Score = {}'.format(m.oob_score_))





def param_tuning(X_train,y_train):

    params = {'randomforestregressor__n_estimators':[10,20,40,60],
                  "randomforestregressor__max_features": randint(10,64),
                  "randomforestregressor__min_samples_split": randint(2, 11),
                  "randomforestregressor__min_samples_leaf": randint(1, 11)
             }

    start = time()
    randomSearch_p1 = RandomizedSearchCV(processing_pipeline1,
                                         param_distributions=params,
                                         n_iter=10,n_jobs=6,
                                         scoring='neg_mean_squared_error'
                                         ).fit(X_train,y_train)

    print('training took {} mins'.format((time() - start)/60.))

    report_best_scores(randomSearch_p1.cv_results_)

    return

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")


def create_pipeline():

    processing_pipeline1 = make_pipeline(RobustScaler(),
                                         StandardScaler(),
                                         RandomForestRegressor())

    processing_pipeline2 = make_pipeline(RobustScaler(),
                                         StandardScaler(),
                                         GradientBoostingRegressor())
