## Machine Learning Models ##
PROJE = "Machine Learning Models "

""" 1- Importing Libraries and Packages """
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import pandas as pd
import numpy as np
import os
import datetime as dt
import time
import warnings
import lightgbm as lgb
import functions as f
f.display()

""" Importing Data"""
test = pd.read_csv("test.csv", parse_dates = ["date"])
train = pd.read_csv("train.csv", parse_dates = ["date"])
sample_sub = pd.read_csv("sample_submission.csv")

""" Explore Data"""
test.head()
train.head()
test.info()
train.info()
train["date"].max()
train["date"].min()
test["date"].max()
test["date"].min()
data = pd.concat([train, test],ignore_index = True)

# Store and Items
data["store"].nunique()
data["item"].nunique()
data.groupby("store")["item"].nunique()
data.isnull().sum()

# Sales and Outlier Detection
std_sales = data["sales"].std()
mean_sales = data["sales"].mean()
plt.hist(data["sales"])
sns.scatterplot(data=data, x="date", y="sales")
sns.boxplot(x=data["sales"])

def check_treshold(df, column):
    q1= df[column].quantile([0.10])[0.10]
    q3 = df[column].quantile([0.90])[0.90]
    iqr =q3-q1
    min_value=q1-1.5*iqr
    max_value=q3+1.5*iqr
    return min_value, max_value
min_value, max_value =  check_treshold(data, "sales")
data = data[(data["sales"]<max_value) | (data["sales"].isna())].reset_index(drop = True)

""" feature engineering """
# Time Columns
def times(df,column):
    dt_column = df[column]
    target_name= column
    attr = ["year", "month", "week", "day", "dayofweek","weekday",
            "is_month_end", "is_month_start"]
    for a in attr:
        df[target_name+"_"+a]=getattr(dt_column.dt,a)

times(data,"date")
data.head()

## Random Noice and Adding mean Columns
def random_noise(df):
    return np.random.normal(scale=1.6, size=(len(df)))

def lag_features(dataframe, lags):
    df = dataframe.copy()
    for lag in lags:
        df['sales_lag_' + str(lag)] = df.groupby(["store", "item"])['sales'].transform(
            lambda x: x.shift(lag))
    return df
lags = [31, 61, 91, 181]
data = lag_features(data, lags)
data.head()
print(data[data.sales.isnull()])

for_na_values = data.groupby(["store","item","date_year","date_month"])["sales"].mean().reset_index()
for_na_values.rename({"sales":"sales_fill"}, axis=1, inplace=True)

data = pd.merge(data, for_na_values, how = "inner", on=["store",  "item" , "date_year",  "date_month"])
data["sales_lag_31"] = np.where(data["sales_lag_31"].isna(),data["sales_fill"],data["sales_lag_31"])+random_noise(data)
data["sales_lag_61"] = np.where(data["sales_lag_61"].isna(),data["sales_fill"],data["sales_lag_61"])+random_noise(data)
data["sales_lag_91"] = np.where(data["sales_lag_91"].isna(),data["sales_fill"],data["sales_lag_91"])+random_noise(data)
data["sales_lag_181"] = np.where(data["sales_lag_181"].isna(),data["sales_fill"],data["sales_lag_181"])+random_noise(data)

def roll_mean_features(dataframe, windows):
    df = dataframe.copy()
    for window in windows:
        df['sales_roll_mean_' + str(window)] = df.groupby(["store", "item"])['sales']. \
                                                          transform(
            lambda x: x.shift(1).rolling(window=window, min_periods=10, win_type="triang").mean())
    return df

windows = [30, 45, 60,90]
data = roll_mean_features(data, windows)
data["sales_roll_mean_30"] = np.where(data["sales_roll_mean_30"].isna(),data["sales_fill"],data["sales_roll_mean_30"])+random_noise(data)
data["sales_roll_mean_45"] = np.where(data["sales_roll_mean_45"].isna(),data["sales_fill"],data["sales_roll_mean_45"])+random_noise(data)
data["sales_roll_mean_60"] = np.where(data["sales_roll_mean_60"].isna(),data["sales_fill"],data["sales_roll_mean_60"])+random_noise(data)
data["sales_roll_mean_90"] = np.where(data["sales_roll_mean_90"].isna(),data["sales_fill"],data["sales_roll_mean_90"])+random_noise(data)

data.head()
print(data.iloc[360:374])


def ewm_features(dataframe, alphas, lags):
    df = dataframe.copy()
    for alpha in alphas:
        for lag in lags:
            df['sales_ewm_alpha_' + str(alpha).replace(".", "") + "_lag_" + str(lag)] = \
                df.groupby(["store", "item"])['sales']. \
                    transform(lambda x: x.shift(lag).ewm(alpha=alpha).mean())
    return df
alphas = [0.95, 0.9, 0.8, 0.7]
lags_2 = [30,45,60,90]

data = ewm_features(data, alphas, lags_2)
data["sales_ewm_alpha_095_lag_30"] = np.where(data["sales_ewm_alpha_095_lag_30"].isna(),data["sales_fill"],data["sales_ewm_alpha_095_lag_30"])+random_noise(data)
data["sales_ewm_alpha_095_lag_45"] = np.where(data["sales_ewm_alpha_095_lag_45"].isna(),data["sales_fill"],data["sales_ewm_alpha_095_lag_45"])+random_noise(data)
data["sales_ewm_alpha_095_lag_60"] = np.where(data["sales_ewm_alpha_095_lag_60"].isna(),data["sales_fill"],data["sales_ewm_alpha_095_lag_60"])+random_noise(data)
data["sales_ewm_alpha_095_lag_90"] = np.where(data["sales_ewm_alpha_095_lag_90"].isna(),data["sales_fill"],data["sales_ewm_alpha_095_lag_90"])+random_noise(data)

data["sales_ewm_alpha_09_lag_30"] = np.where(data["sales_ewm_alpha_09_lag_30"].isna(),data["sales_fill"],data["sales_ewm_alpha_09_lag_30"])+random_noise(data)
data["sales_ewm_alpha_09_lag_45"] = np.where(data["sales_ewm_alpha_09_lag_45"].isna(),data["sales_fill"],data["sales_ewm_alpha_09_lag_45"])+random_noise(data)
data["sales_ewm_alpha_09_lag_60"] = np.where(data["sales_ewm_alpha_09_lag_60"].isna(),data["sales_fill"],data["sales_ewm_alpha_09_lag_60"])+random_noise(data)
data["sales_ewm_alpha_09_lag_90"] = np.where(data["sales_ewm_alpha_09_lag_90"].isna(),data["sales_fill"],data["sales_ewm_alpha_09_lag_90"])+random_noise(data)

data["sales_ewm_alpha_08_lag_30"] = np.where(data["sales_ewm_alpha_08_lag_30"].isna(),data["sales_fill"],data["sales_ewm_alpha_08_lag_30"])+random_noise(data)
data["sales_ewm_alpha_08_lag_45"] = np.where(data["sales_ewm_alpha_08_lag_45"].isna(),data["sales_fill"],data["sales_ewm_alpha_08_lag_45"])+random_noise(data)
data["sales_ewm_alpha_08_lag_60"] = np.where(data["sales_ewm_alpha_08_lag_60"].isna(),data["sales_fill"],data["sales_ewm_alpha_08_lag_60"])+random_noise(data)
data["sales_ewm_alpha_08_lag_90"] = np.where(data["sales_ewm_alpha_08_lag_90"].isna(),data["sales_fill"],data["sales_ewm_alpha_08_lag_90"])+random_noise(data)

data["sales_ewm_alpha_07_lag_30"] = np.where(data["sales_ewm_alpha_07_lag_30"].isna(),data["sales_fill"],data["sales_ewm_alpha_07_lag_30"])+random_noise(data)
data["sales_ewm_alpha_07_lag_45"] = np.where(data["sales_ewm_alpha_07_lag_45"].isna(),data["sales_fill"],data["sales_ewm_alpha_07_lag_45"])+random_noise(data)
data["sales_ewm_alpha_07_lag_60"] = np.where(data["sales_ewm_alpha_07_lag_60"].isna(),data["sales_fill"],data["sales_ewm_alpha_07_lag_60"])+random_noise(data)
data["sales_ewm_alpha_07_lag_90"] = np.where(data["sales_ewm_alpha_07_lag_90"].isna(),data["sales_fill"],data["sales_ewm_alpha_07_lag_90"])+random_noise(data)
data.shape
data.drop(columns = "sales_fill", inplace = True)


original = data.copy()

# One-Hot Enncoding
data = pd.get_dummies(data, columns=['store', 'item', 'date_dayofweek', 'date_month',"date_day","date_weekday","date_year","date_week"])
print(data.columns)
#Logaritmic Transformation
data['sales'] = np.log1p(data["sales"].values)  #targetla ilgli bir mesele
data.head(3)

# Split train-validation
train = data[data["date"]<"2017-01-01"]
train.head(3)
train.date.min()
train.date.max()
validation = data[(data["date"]>="2017-01-01" )& (data["date"]<= "2017-03-31")]
validation.date.min()
validation.date.max()

cols = [col for col in train.columns if col not in ['date', 'id', "sales"]]
Y_train = train["sales"]
X_train = train[cols]
Y_val = validation['sales']
X_val = validation[cols]
print(Y_train.shape, X_train.shape, Y_val.shape, X_val.shape)

import math
def rmse(x,y):
    return math.sqrt(((x-y)**2).mean())

def print_score(m):
    print(f"RMSE of train set {rmse(m.predict(X_train), Y_train)}")
    print(f"RMSE of validation set {rmse(m.predict(X_val), Y_val)}")
    print(f"R^2 of train set {m.score(X_train, Y_train)}")
    print(f"R^2 of validation set {m.score(X_val, Y_val)}")

from sklearn.ensemble import RandomForestRegressor

def get_sample(df,n):
    idxs = np.random.permutation(len(df))[:n]
    return idxs, df.iloc[idxs].copy()

idxs, x_train = get_sample(X_train,3000)
y_train = Y_train.iloc[idxs]

m = RandomForestRegressor(n_estimators=10, n_jobs=-1)
start_time = time.time ()
m.fit(x_train,y_train)
end_time = time.time ()
print(end_time - start_time, "seconds")
print_score(m)

m = RandomForestRegressor(n_estimators=30, n_jobs=-1)
start_time = time.perf_counter ()
m.fit(x_train,y_train)
end_time = time.perf_counter ()
print(end_time - start_time, "seconds")
print_score(m)

m = RandomForestRegressor(n_estimators=100, n_jobs=-1)
start_time = time.perf_counter ()
m.fit(x_train,y_train)
end_time = time.perf_counter ()
print(end_time - start_time, "seconds")
print_score(m)

feature_score = pd.Series(m.feature_importances_, index=X_train.columns).sort_values(ascending = False)
sns.barplot(feature_score, feature_score.index)
feature_score.head(10)
type(feature_score.head())
importants =  feature_score.head(100).index
print(importants)

Y_train = train["sales"]
X_train_new = train[importants]
Y_val = validation['sales']
X_val_new = validation[importants]
print(Y_train.shape, X_train_new.shape, Y_val.shape, X_val_new.shape)

idxs, x_train_new = get_sample(X_train_new,3000)
y_train_new = Y_train.iloc[idxs]
def print_score_new(m):
    print(f"RMSE of train set {rmse(m.predict(X_train_new), Y_train)}")
    print(f"RMSE of validation set {rmse(m.predict(X_val_new), Y_val)}")
    print(f"R^2 of train set {m.score(X_train_new, Y_train)}")
    print(f"R^2 of validation set {m.score(X_val_new, Y_val)}")

new = RandomForestRegressor(n_estimators=100,max_features =0.5 ,n_jobs=-1)
start_time = time.time ()
new.fit(x_train_new,y_train_new)
end_time = time.time ()
print(end_time - start_time, "seconds")
print_score_new(new)



""" 6- Model Fitting and Predicting """
# sklearn Models to Test

from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import make_scorer, accuracy_score
from sklearn.model_selection import GridSearchCV



