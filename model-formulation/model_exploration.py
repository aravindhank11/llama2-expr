import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import pandas as pd
import numpy as np
from collections import Counter
import math
import statistics
from imblearn.over_sampling import SMOTE
from time import time

# Sklearn models
from sklearn.linear_model import LogisticRegression, LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR

# Sklearn helper functions
from sklearn.model_selection import train_test_split
from sklearn.metrics import log_loss, mean_squared_error, r2_score, make_scorer, accuracy_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, KFold, cross_val_score, GridSearchCV

# Boosting models
import xgboost as xgb
from catboost import CatBoostClassifier, CatBoostRegressor
import lightgbm as lgbm

def preprocess_dataset_classification_hol(job_size):
    df = pd.read_csv(f'./data/tiebreaker-training-set-{job_size}-hol.csv')

    # Convert label
    label_conversion = {'no_hol': 0, 'hol': 1}
    df['label'] = df['label'].map(label_conversion)
    
    # Drop a few columns we don't need
    columns_to_drop = [col for col in df.columns for prefix in ['model_load_time', 'image_load_time', 'model_size'] if col.startswith(prefix)]
    df.drop(columns=columns_to_drop, inplace=True)
    model_name_columns = []
    for i in range (1, job_size + 1):
        model_name_columns.append(f'm_{i}')
    df.drop(columns=model_name_columns, inplace=True)
    df.drop(columns=['exp_gpu_no'], inplace=True)

    # Create train and test set manually to do SMOTE oversampling of minor classes
    df_train = df.sample(frac=0.75, random_state=100)
    df_test = df.drop(df_train.index)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    train_features = df_train.drop(columns=['label'], axis=1)
    train_target = df_train['label']
    oversample = SMOTE(random_state=42)
    train_features, train_target = oversample.fit_resample(train_features, train_target)

    return train_features, train_target, df_test

def preprocess_dataset_classification(job_size, optimization_metric):
    df = pd.read_csv(f'./data/tiebreaker-training-set-{job_size}.csv')

    # Convert label
    label_conversion = {'mps': 0, 'ts': 1, 'mig': 2}
    if optimization_metric == 'max':
        df['label'] = df['top_max_mech'].map(label_conversion)
    elif optimization_metric == 'median':
        df['label'] = df['top_median_mech'].map(label_conversion)
    
    # Drop few columns not required
    columns_to_drop = [col for col in df.columns for prefix in ['model_load_time', 'image_load_time', 'model_size'] if col.startswith(prefix)]
    df.drop(columns=columns_to_drop, inplace=True)

    model_name_columns = []
    for i in range (1, job_size + 1):
        model_name_columns.append(f'm_{i}')
    df.drop(columns=model_name_columns, inplace=True)
    df.drop(columns=['top_max_mech', 'top_median_mech', 'mps_max_slowdown', 'mps_median_slowdown', 'ts_max_slowdown', 'ts_median_slowdown', 'mig_max_slowdown', 'mig_median_slowdown', 'exp_gpu_no'], inplace=True)
    
    # Create train and test set manually to do SMOTE oversampling of minor classes
    df_train = df.sample(frac=0.75, random_state=42)
    df_test = df.drop(df_train.index)
    df_train.reset_index(drop=True, inplace=True)
    df_test.reset_index(drop=True, inplace=True)
    train_features = df_train.drop(columns=['label'], axis=1)
    train_target = df_train['label']
    oversample = SMOTE(random_state=42)
    train_features, train_target = oversample.fit_resample(train_features, train_target)

    return train_features, train_target, df_test
    
def final_preds_ci(test_preds, probas_list, alpha):
    # Confidence interval logic
    final_predictions = []
    for i, prediction in enumerate(test_preds):
        if prediction == 0:
            if probas_list[i] >= alpha:
                final_predictions.append(prediction)
            else:
                final_predictions.append(2)
        else:
            final_predictions.append(prediction)
    return final_predictions

def report_accuracy(test_preds, test_target, model):
    # Compute Accuracy
    no_same = 0
    for index, val in enumerate(test_preds):
        if val == test_target[index]:
            no_same += 1
    accuracy = no_same / len(test_preds) * 100
    print(f'{model} Accuracy: {accuracy}')

    no_mps = 0
    no_ts = 0
    no_mig = 0
    c = Counter(test_target)
    for item in c.items():
        if item[0] == 0:
            no_mps = item[1]
        elif item[0] == 1:
            no_ts = item[1]
        else:
            no_mig = item[1]

    mps_correct = 0
    ts_correct = 0
    mig_correct = 0
    for index, value in enumerate(test_preds):
        if value == test_target[index]:
            if value == 0:
                mps_correct += 1
            elif value == 1:
                ts_correct += 1
            elif value == 2:
                mig_correct += 1
    mps_accuracy = mps_correct / no_mps * 100
    ts_accuracy = ts_correct / no_ts * 100
    mig_accuracy = mig_correct / no_mig * 100
    print(f'{model} Accuracy Breakdown:')
    print('MPS: {} / {} = {}'.format(mps_correct, no_mps, mps_accuracy ))
    print('TS: {} / {} = {}'.format(ts_correct, no_ts, ts_accuracy ))
    print('MIG: {} / {} = {}'.format(mig_correct, no_mig, mig_accuracy ))
    print()

def report_hol_accuracy(test_preds, test_target, model):
    # Compute Accuracy
    no_same = 0
    for index, val in enumerate(test_preds):
        if val == test_target[index]:
            no_same += 1
    accuracy = no_same / len(test_preds) * 100
    print(f'{model} Accuracy: {accuracy}')

    
'''
Classifiers
'''
def rf_class_tuning(optimization_metric, alpha):
    
    # Get train and test data ready to use
    # train_features, train_target, df_test = preprocess_dataset_classification(job_size=3, optimization_metric=optimization_metric)
    train_features, train_target, df_test = preprocess_dataset_classification_hol(job_size=3)
    print(train_features)
    test_features = df_test.drop(columns=['label'], axis=1)
    test_target = df_test['label']

    # Train model
    # rf_model = RandomForestClassifier(random_state=5, criterion='log_loss', max_depth=11, max_features=5, min_samples_leaf=1, min_samples_split=2, n_estimators=300)
    # rf_model = RandomForestClassifier(random_state=5, criterion='log_loss', max_depth=11, max_features=5, min_samples_leaf=1, min_samples_split=2, n_estimators=800)
    rf_model = RandomForestClassifier(random_state=5, criterion='log_loss', max_depth=5, max_features=5, min_samples_leaf=1, min_samples_split=2, n_estimators=75)
    train_start = time()
    rf_model.fit(train_features, train_target)
    print(f'Time to train: {time() - train_start}')

    # Model predictions
    test_probas = rf_model.predict_proba(test_features)
    probas_list = []
    for probas in test_probas:
        probas_list.append(max(probas))
    train_preds = list(rf_model.predict(train_features))
    test_preds = list(rf_model.predict(test_features))
    test_target = list(test_target)
    df_test['predictions'] = test_preds
    df_test['prediction_probas'] = probas_list
    df_test['ground_truths'] = test_target
    df_test['final_predictions'] = final_preds_ci(test_preds, probas_list, alpha)

    # Accuracy
    # report_accuracy(test_preds, test_target, 'Random Forest')
    report_hol_accuracy(train_preds, train_target, 'Random Forest Train')
    report_hol_accuracy(test_preds, test_target, 'Random Forest Test')

def cb_class_tuning(optimization_metric, alpha):
    # Get train and test data ready to use
    # train_features, train_target, df_test = preprocess_dataset_classification(job_size=3, optimization_metric=optimization_metric)
    train_features, train_target, df_test = preprocess_dataset_classification_hol(job_size=3)
    test_features = df_test.drop(columns=['label'], axis=1)
    test_target = df_test['label']

    # Train model
    cb = CatBoostClassifier(iterations=300, learning_rate = 0.07, depth=11, verbose=False, random_state=42) 
    train_start = time()
    cb.fit(train_features, train_target)
    print(f'Time to train: {time() - train_start}')

    # Model predictions
    test_probas = cb.predict_proba(test_features)
    probas_list = []
    for probas in test_probas:
        probas_list.append(max(probas))
    test_preds = list(cb.predict(test_features))
    test_target = list(test_target)
    df_test['predictions'] = test_preds
    df_test['prediction_probas'] = probas_list
    df_test['ground_truthes'] = test_target
    df_test['final_predictions'] = final_preds_ci(test_preds, probas_list, alpha)

    # Accuracy
    # report_accuracy(test_preds, test_target, 'CatBoost')
    report_hol_accuracy(test_preds, test_target, 'CatBoost')

def xgb_class_tuning(optimization_metric, alpha):
    # Get train and test data ready to use
    train_features, train_target, df_test = preprocess_dataset_classification(job_size=3, optimization_metric=optimization_metric)
    test_features = df_test.drop(columns=['label'], axis=1)
    test_target = df_test['label']

    # Train model
    # xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, colsample_bytree=0.5, gamma=0.2, learning_rate=0.01, max_depth=4, subsample=0.6, seed=42)
    xgb_model = xgb.XGBClassifier(objective='multi:softmax', max_depth=50, learning_rate=0.1)
    train_start = time()
    xgb_model.fit(train_features, train_target)
    print(f'Time to train: {time() - train_start}')

    # Model predictions
    test_probas = xgb_model.predict_proba(test_features)
    probas_list = []
    for probas in test_probas:
        probas_list.append(max(probas))
    test_preds = list(xgb_model.predict(test_features))
    test_target = list(test_target)
    df_test['predictions'] = test_preds
    df_test['prediction_probas'] = probas_list
    df_test['ground_truthes'] = test_target
    df_test['final_predictions'] = final_preds_ci(test_preds, probas_list, alpha)

    # Accuracy
    report_accuracy(test_preds, test_target, 'XGBoost')


if __name__=='__main__':
    rf_class_tuning(optimization_metric='max', alpha=0.60)
    # cb_class_tuning(optimization_metric='max', alpha=0.60)
    # xgb_class_tuning(optimization_metric='max', alpha=0.60)