import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math

import pandas as pd
from catboost import CatBoostRegressor

import os
import glob

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


def parsing_data():
    os.chdir(r"data\raw_train")

    files = glob.glob('*.csv')

    combined = pd.DataFrame()
    n = 0

    for file in files:
        data = pd.read_csv(file)
        combined = pd.concat([combined, data])
        n += 1
        if n == 250:
            break

    os.chdir("..\..")
    combined.to_csv("combined_csv.csv", index=False, encoding='utf-8-sig')


def preprocessing_data():
    df = pd.read_csv('combined_csv.csv', delimiter=',')

    # Выберем только интересующие нас столбцы
    df = df[['SK_Well', 'SK_Calendar', 'GRPCount', 'WeightedParticlesFactor_mg_l', 'ProducingGOR_m3_t',
             'MeasureMRM17', 'MeasureMRM205', 'MeasureMRM142', 'MeasureOIS1', 'MeasureOIS2', 'MeasureOIS3',
             'WellHeadPressure_atm', 'Watercut_t', 'Watercut_m3', 'Watercut', 'OilViscosity_cps',
             'ActivePowerCS_kWt', 'FailuresCountFromLastWellWork', 'daysFromLastStart', 'daysToFailure', ]]

    # заменяем все значения NaN на ноль, чтобы не было ошибок при обучении модели; перетасовываем данные
    df = df.fillna(0)
    df = shuffle(df).reset_index(drop=True)
    return df


def train(df):
    # Выбираем данные для обучения и метки
    X = np.array(df.iloc[:, 2:19])
    y = df['daysToFailure']

    # разделяем данные для тренировки и валидации
    X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.80, random_state=42)

    class RMSLE(object):
        def calc_ders_range(self, approxes, targets, weights):
            assert len(approxes) == len(targets)
            if weights is not None:
                assert len(weights) == len(approxes)

            result = []
            for index in range(len(targets)):
                val = max(approxes[index], 0)
                der1 = math.log1p(targets[index]) - math.log1p(max(0, approxes[index]))
                der2 = -1 / (max(0, approxes[index]) + 1)

                if weights is not None:
                    der1 *= weights[index]
                    der2 *= weights[index]

                result.append((der1, der2))
            return result

    class RMSLE_val(object):
        def get_final_error(self, error, weight):
            return np.sqrt(error / (weight + 1e-38))

        def is_max_optimal(self):
            return False

        def evaluate(self, approxes, target, weight):
            assert len(approxes) == 1
            assert len(target) == len(approxes[0])

            approx = approxes[0]

            error_sum = 0.0
            weight_sum = 0.0

            for i in range(len(approx)):
                w = 1.0 if weight is None else weight[i]
                weight_sum += w
                error_sum += w * ((math.log1p(max(0, approx[i])) - math.log1p(max(0, target[i]))) ** 2)

            return error_sum, weight_sum

    # Создаем модель и задаем характеристики
    model = CatBoostRegressor(iterations=3000,
                              early_stopping_rounds=100,
                              grow_policy='Depthwise',
                              depth=10,
                              loss_function=RMSLE(),
                              l2_leaf_reg=1,
                              learning_rate=0.1,
                              eval_metric=RMSLE_val())

    # тренируем модель
    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        plot=True)

    os.chdir('model')
    model.save_model('model.dump')
    os.chdir("..")


def get_sample_test():
    os.chdir('data\sample_test')
    files = os.listdir()
    os.chdir("..\..")
    return files


def predict():
    files = get_sample_test()

    os.chdir('model')
    model = CatBoostRegressor()
    model.load_model('model.dump')
    os.chdir('..')

    lst_file = []
    lst_predicted = []

    for file in files:
        test = pd.read_csv(file, delimiter=',')

        # Выберем только интересующие нас столбцы
        test = test[['SK_Well', 'SK_Calendar', 'GRPCount', 'WeightedParticlesFactor_mg_l', 'ProducingGOR_m3_t',
                     'MeasureMRM17', 'MeasureMRM205', 'MeasureMRM142', 'MeasureOIS1', 'MeasureOIS2', 'MeasureOIS3',
                     'WellHeadPressure_atm', 'Watercut_t', 'Watercut_m3', 'Watercut', 'OilViscosity_cps',
                     'ActivePowerCS_kWt', 'FailuresCountFromLastWellWork', 'daysFromLastStart']]

        # заменяем все значения NaN на ноль, чтобы не было ошибок при обучении модели
        test = test.fillna(0)

        X = np.array(test.iloc[:, 2:18])
        predict = model.predict(X)

        lst_file.append(file)
        lst_predicted.append(predict[-1])

    dict_files = {'filename': lst_file, 'daysToFailure': lst_predicted}
    df = pd.DataFrame(data=dict_files)
    os.mkdir('output')
    df.to_csv('output\submission.csv', index=False)


def main():
    parsing_data()
    df = preprocessing_data()
    train(df)
    predict()


if __name__ == '__main__':
    main()
