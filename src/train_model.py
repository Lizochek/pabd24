# """Train model and save checkpoint"""
#
# import argparse
# import logging
# import pandas as pd
# from sklearn.linear_model import LinearRegression
# from sklearn.metrics import mean_absolute_error
# from joblib import dump
#
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     filename='log/train_model.log',
#     encoding='utf-8',
#     level=logging.DEBUG,
#     format='%(asctime)s %(message)s')
#
# TRAIN_DATA = 'data/proc/train.csv'
# VAL_DATA = 'data/proc/val.csv'
# MODEL_SAVE_PATH = 'models/linear_regression_v01.joblib'
#
#
# def main(args):
#     df_train = pd.read_csv(TRAIN_DATA)
#     x_train = df_train[['total_meters']]
#     y_train = df_train['price']
#     df_val = pd.read_csv(VAL_DATA)
#     x_val = df_val[['total_meters']]
#     y_val = df_val['price']
#
#     linear_model = LinearRegression()
#     linear_model.fit(x_train, y_train)
#     dump(linear_model, args.model)
#     logger.info(f'Saved to {args.model}')
#
#     r2 = linear_model.score(x_train, y_train)
#     y_pred = linear_model.predict(x_val)
#     mae = mean_absolute_error(y_pred, y_val)
#     c = int(linear_model.coef_[0])
#     inter = int(linear_model.intercept_)
#
#     logger.info(f'R2 = {r2:.3f}     MAE = {mae:.0f}     Price = {c} * area + {inter}')
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-m', '--model',
#                         help='Model save path',
#                         default=MODEL_SAVE_PATH)
#     args = parser.parse_args()
#     main(args)
# # """Train model and save checkpoint"""
# #
# # import argparse
# # import logging
# # import pandas as pd
# # import xgboost as xgb
# # from sklearn.linear_model import LinearRegression
# # from sklearn.metrics import mean_absolute_error, r2_score
# # from joblib import dump
# #
# # logger = logging.getLogger(__name__)
# # logging.basicConfig(
# #     filename='log/train_model.log',
# #     encoding='utf-8',
# #     level=logging.DEBUG,
# #     format='%(asctime)s %(message)s')
# #
# # TRAIN_DATA = 'data/proc/train.csv'
# # VAL_DATA = 'data/proc/val.csv'
# # MODEL_SAVE_PATH = 'models/linear_regression_v01.joblib'
# #
#
# # def main(args):
# #     df_train = pd.read_csv(TRAIN_DATA)
# #     x_train = df_train[['total_meters']]
# #     y_train = df_train['price']
# #     df_val = pd.read_csv(VAL_DATA)
# #     x_val = df_val[['total_meters']]
# #     y_val = df_val['price']
# #
# #     # linear_model = LinearRegression()
# #     # linear_model.fit(x_train, y_train)
# #     # dump(linear_model, args.model)
# #     # logger.info(f'Saved to {args.model}')
# #     #
# #     # r2 = linear_model.score(x_train, y_train)
# #     # y_pred = linear_model.predict(x_val)
# #     # mae = mean_absolute_error(y_pred, y_val)
# #     # c = int(linear_model.coef_[0])
# #     # inter = int(linear_model.intercept_)
# #
# #     # Инициализация и обучение модели XGBoost
# #     xgb_model = xgb.XGBRegressor(objective='reg:squarederror', colsample_bytree=0.3, learning_rate=0.1,
# #                                  max_depth=5, alpha=10, n_estimators=100)
# #     xgb_model.fit(x_train, y_train)
# #
# #     # Сохранение модели
# #     dump(xgb_model, args.model)
# #     logger.info(f'Saved to {args.model}')
# #
# #     # Оценка модели
# #     y_pred = xgb_model.predict(x_val)
# #     mae = mean_absolute_error(y_val, y_pred)
# #     r2 = r2_score(y_val, y_pred)
# #
# #     # Расчет коэффициентов для формулы цены
# #     c = xgb_model.feature_importances_[0]  # Используем важность признака как коэффициент
# #     inter = y_train.mean()  # Используем среднюю цену как базовое смещение
# #
# #     logger.info(f'R2 = {r2:.3f}     MAE = {mae:.0f}     Price = {c:.3f} * area + {inter:.0f}')
# #
# #
# # if __name__ == '__main__':
# #     parser = argparse.ArgumentParser()
# #     parser.add_argument('-m', '--model',
# #                         help='Model save path',
# #                         default=MODEL_SAVE_PATH)
# #     args = parser.parse_args()
# #     main(args)
# src/train_model.py
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
VAL_DATA = 'data/proc/val.csv'
MODEL_SAVE_PATH = 'models/random_forest_v01.joblib'

def main(args):
    df_train = pd.read_csv(TRAIN_DATA)
    x_train = df_train[['floor', 'rooms_count', 'total_meters']]
    y_train = df_train['price']
    df_val = pd.read_csv(VAL_DATA)
    x_val = df_val[['floor', 'rooms_count', 'total_meters']]
    y_val = df_val['price']

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

    logger.info(f'R2 = {r2:.3f}     MAE = {mae:.0f}')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='Model save path',
                        default=MODEL_SAVE_PATH)
    args = parser.parse_args()
    main(args)
