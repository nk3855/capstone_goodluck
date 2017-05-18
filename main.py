# -*- coding: utf-8 -*-
import sklearn.linear_model.logistic as logistic
import sklearn.ensemble as randomforest
import sklearn.svm as svm
import sklearn.preprocessing as sc
from sklearn.model_selection import KFold
import glob
import numpy as np
import datetime
import pandas as pd
import matplotlib.pyplot as plt

def do_logistic_regression(x_train, y_train):
    classifier = logistic.LogisticRegression()
    classifier.fit(x_train, y_train)

    return classifier

def do_random_forest(x_train, y_train):
    classifier = randomforest.RandomForestClassifier()
    classifier.fit(x_train, y_train)

    return classifier

def do_svm(x_train, y_train):
    classifier = svm.SVC()
    classifier.fit(x_train, y_train)

    return classifier

def make_dataset(df, time_lags):
    df_lag = pd.DataFrame(index=df.index)
    df_lag["Close"] = df["Close"]

    df_lag["Close_Lag%s" % str(time_lags)] = df["Close"].shift(time_lags)
    df_lag["Close_Lag%s_Change" % str(time_lags)] = df_lag["Close_Lag%s" % str(time_lags)].pct_change() * 100.0

    df_lag["Close_Direction"] = np.sign(df_lag["Close_Lag%s_Change" % str(time_lags)])

    return df_lag.dropna(how='any')

def split_dataset(df, input_column_array, output_column, spllit_ratio):
    split_date = get_date_by_percent(df.index[0], df.index[df.shape[0] - 1], spllit_ratio)

    input_data = df[input_column_array]
    output_data = df[output_column]

    X_train = input_data[input_data.index < split_date]
    X_test = input_data[input_data.index >= split_date]
    Y_train = output_data[output_data.index < split_date]
    Y_test = output_data[output_data.index >= split_date]

    return X_train, X_test, Y_train, Y_test

def get_date_by_percent(start_date, end_date, percent):
    days = (end_date - start_date).days
    target_days = np.trunc(days * percent)
    target_date = start_date + datetime.timedelta(days=target_days)

    return target_date

def test_classifier(classifier, test_value):
    pred = classifier.predict(test_value)

    return pred[0]

def checkUpDown(lastPrice, todayPrice):
    if lastPrice < todayPrice:
        return 1
    elif lastPrice > todayPrice:
        return -1
    else:
        return 0

if __name__ == "__main__":
    file_list = glob.glob("/Users/Eun/Documents/out/*.csv")
    index = ["High", "Low", "Open", "Close", "Volume"]

    total_len = len(file_list)
    lrCount = 0
    svmCount = 0
    rfCount = 0

    for file_name in file_list:
        file = pd.DataFrame.from_csv(file_name, header=None)
        file.columns = index

        lastPrice = file.ix[-2, -2]
        todayPrice = file.ix[-1, -2]

        dataset = make_dataset(file.ix[0:-3, :], 1)
        datelist = dataset.index
        X_train_date = []
        X_train_value = []
        Y_train_value = []

        kf = KFold(n_splits=10)
        for train, test in kf.split(dataset):
            for row in train:
                X_train_date.append(datelist[row])
                X_train_value.append(dataset.iloc[row, 1])
                Y_train_value.append(dataset.iloc[row, 3])

            X_train = pd.DataFrame(index=X_train_date)
            X_train["Close"] = X_train_value
            X_train["Close_Lag1_Change"] = Y_train_value

        lr_classifier = do_logistic_regression(X_train.ix[:, 0:1], X_train["Close_Lag1_Change"])
        lr_pred = test_classifier(lr_classifier, lastPrice)

        if checkUpDown(lastPrice, todayPrice) == lr_pred:
            lrCount += 1

        svm_classifier =do_svm(X_train.ix[:, 0:1], X_train["Close_Lag1_Change"])
        svm_pred = test_classifier(svm_classifier, lastPrice)
        if checkUpDown(lastPrice, todayPrice) == svm_pred:
            svmCount += 1

        rf_classifier = do_random_forest(X_train.ix[:, 0:1], X_train["Close_Lag1_Change"])
        rf_pred = test_classifier(rf_classifier, lastPrice)
        if checkUpDown(lastPrice, todayPrice) == rf_pred:
            rfCount += 1

        print(file_name)

    print('lr = %0.2f, svm = %0.2f, rf = %0.2f' % (float(rfCount) / float(total_len), float(svmCount) / float(total_len), float(rfCount) / float(total_len)))
