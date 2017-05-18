import pandas as pd
import pandas_datareader.data as web
from pandas.tools.plotting import scatter_matrix, autocorrelation_plot

import datetime
import matplotlib.pyplot as plt
import numpy as np

from sklearn import linear_model, decomposition
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import LinearSVC

def download_stock_data(file_name, company_code, year1, month1, date1, year2, month2, date2):
    start = datetime.datetime(year1, month1, date1)  # 시작 날짜 데이터 지정
    end = datetime.datetime(year2, month2, date2)  # 마지막 날짜 데이터 지정
    df = web.DataReader("%s.KS" % company_code, "yahoo", start, end)  # yahoo finance 데이터 가져오기
    return df


df_samsung = download_stock_data('samsung.data', '005930', 2015, 1, 1, 2015, 11, 30)
df_hanmi = download_stock_data('hanmi.data', '128940', 2015, 1, 1, 2015, 11, 30)


## 데이터셋 만들기 : 인자로 전달된 데이터프레임을 바탕으로 학습과 테스트에 사용할 데이터프레임을 만들어 돌려준다
def make_dataset(df, time_lags=5):  # time_lag : 현재일을 기준으로 몇일전의 데이터를 사용할 것인가
    df_lag = pd.DataFrame(index=df.index)
    df_lag['Close'] = df['Close']
    df_lag['Volume'] = df['Volume']

    df_lag['Close_Lag%s' % str(time_lags)] = df['Close'].shift(time_lags)
    df_lag['Close_Lag%s_Change' % str(time_lags)] = df_lag['Close_Lag%s' % str(
        time_lags)].pct_change() * 100.0  # pct_change : 주어진 데이터의 변화를 퍼센트로 계산

    df_lag['Volume_Lag%s' % str(time_lags)] = df['Volume'].shift(time_lags)
    df_lag['Volume_Lag%s_Change' % str(time_lags)] = df_lag['Volume_Lag%s_Change' % str(time_lags)]

    df_lag['Close_Direction'] = np.sign(
        df_lag['Close_Lag%s_Change' % str(time_lags)])  # Direction 변수는 데이터의 방향을 나타낸다. 양수면 상승, 음수면 하락
    df_lag['Volume_Direction'] = np.sign(df_lag['Volume_Lag%s_Change' % str(time_lags)])
    return df_lag.dropna(how='any')


## 데이터셋 나누기
## input_column_array : 입력변수 Dataframe을 배열형태로 전달
## output_column : 출력변수
def split_dataset(df_lag, input_column_array, output_column, split_ratio):
    split_date = get_date_by_percent(df_lag.index[0], df_lag.index[df.shape[0] - 1], split_ratio)

    input_data = df_lag[input_column_array]
    output_data = df_lag[output_column]
    ## X_train : 학습에 사용할 입력변수
    ## Y_train : 학습에 사용할 출력변수
    ## X_test : 테스트에 사용할 입력변수
    ## Y_test : 테스트에 사용할 출력변수
    X_train = input_data[input_data.index < split_date]
    X_test = input_data[input_data.index >= split_date]
    Y_train = output_data[output_data.index < split_date]
    Y_test = output_data[output_data.index >= split_date]

    return X_train, X_test, Y_test, Y_train


def get_date_by_percent(start_date, end_date, percent):
    days = (end_date - start_date).days
    target_days = np.trunc(days * percent)
    target_date = start_date + datetime.timedelta(days=target_days)
    return target_date


#####################################################################################################
## 주가방향 예측변수 작성
def do_logistic_regression(x_train, y_train):
    classifier = linear_model.LogisticRegression()
    classifier.fit(x_train, y_train)
    return classifier


def do_random_forest(x_train, y_train):
    classifier = RandomForestRegressor()
    classifier.fit(x_train, y_train)
    return classifier


def do_svm(x_train, y_train):
    classifier = LinearSVC()
    classifier.fit(x_train, y_train)
    return classifier


def test_classifieer(classifier, x_test, y_test, ):
    pred = classifier.predict(x_test)

    hit_count = 0
    total_count = len(y_test)
    for index in range(total_count):
        if (pred[index]) == (y_test[index]):
            hit_count = hit_count + 1
    hit_ratio = hit_count / total_count
    score = classifier.score(x_test, y_test)

    return hit_ratio, score


if __name__ == "__main__":
    for time_lags in range(1, 6):
        print("- Time Lage = %s" % (time_lags))

        for company in [df_samsung, df_hanmi]:
            df_dataset = make_dataset(company, time_lags)
            X_train, X_test, Y_train, Y_test = split_dataset(df_dataset, ['Close_Lag%s(time_lags)'], 'Close_Direction',
                                                             0.75)

            lr_classifier = do_logistic_regression(X_train, Y_train)
            lr_hit_ratio, lr_score = test_classifieer(lr_classifier, X_test, Y_test)

            rf_classifier = do_random_forest(X_train, Y_train)
            rf_hit_ratio, rf_score = test_classifieer(rf_classifier, X_test, Y_test)

            svm_classifier = do_svm(X_train, Y_train)
            svm_hit_ratio, svm_score = test_classifieer(svm_classifier, X_test, Y_test)


            #print(" %s : Hit Ratio - Logistic Regression=%0.2f, RandomForest=%0.2f, SVM=%0.2f" % (company, lr_hit_ratio, rf_classifier, svm_hit_ratio))
