#src/train_model.py
import argparse
import logging
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from joblib import dump

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/train_model.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s')

TRAIN_DATA = 'data/proc/train.csv'
VAL_DATA = 'data/proc/test.csv'
MODEL_SAVE_PATH = 'models/linear_regression_ff_v01.joblib'

def main(args):
    df_train = pd.read_csv(TRAIN_DATA)
    # Преобразование столбцов в целые числа
    floor = df_train['floor'].astype(int)
    floors_count = df_train['floors_count'].astype(int)

    # Определение признаков is_first и is_last
    is_first = (floor == 1).astype(int)
    is_last = (floor == floors_count).astype(int)

    # Создание DataFrame с признаками для обучения
    x_train = df_train[['total_meters', 'floor', 'floors_count', 'rooms_count']].copy()
    x_train.loc[:, 'is_first'] = is_first
    x_train.loc[:, 'is_last'] = is_last

    # Целевая переменная
    y_train = df_train['price']

    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)
    dump(model, args.model)
    logger.info(f'Saved to {args.model}')

    r2 = model.score(x_train, y_train)
    y_pred = model.predict(x_val)
    mae = mean_absolute_error(y_pred, y_val)

    # Выводим важность признаков
    feature_importances = model.feature_importances_
    logger.info(f'Feature importances: {feature_importances}')

    logger.info(f'R2 = {r2:.3f} ')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='Model save path',
                        default=MODEL_SAVE_PATH)
    args = parser.parse_args()
    main(args)

# import argparse
# import logging
# import pandas as pd
# from joblib import dump
# from xgboost import XGBRegressor

# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     filename='log/train_model.log',
#     encoding='utf-8',
#     level=logging.DEBUG,
#     format='%(asctime)s %(message)s')
#
# TRAIN_DATA = 'data/proc/train.csv'
# VAL_DATA = 'data/proc/val.csv'
# MODEL_SAVE_PATH = 'models/xgboost_model_v01.joblib'
#
#
# def main(args):
#     df_train = pd.read_csv(TRAIN_DATA)
#     x_train = df_train[['floor', 'rooms_count', 'total_meters']]
#     y_train = df_train['price']
#
#     xgboost_model = XGBRegressor()
#     xgboost_model.fit(x_train, y_train)
#     dump(xgboost_model, args.model)
#     logger.info(f'Saved to {args.model}')
#
#     r2 = xgboost_model.score(x_train, y_train)
#     logger.info(f'R2 = {r2:.3f}')
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-m', '--model',
#                         help='Model save path',
#                         default=MODEL_SAVE_PATH)
#     args = parser.parse_args()
#     main(args)
