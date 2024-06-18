import argparse
import logging
import pandas as pd
from joblib import load
from sklearn.metrics import mean_absolute_error

MODEL_SAVE_PATH = 'models/linear_regression_v01.joblib'
TEST_DATA = 'data/proc/test.csv'

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/test_model.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s')


def main(args):
    df_test = pd.read_csv(TEST_DATA)
    # Преобразование столбцов в целые числа
    floor = df_test['floor'].astype(int)
    floors_count = df_test['floors_count'].astype(int)

    # Определение признаков is_first и is_last
    is_first = (floor == 1).astype(int)
    is_last = (floor == floors_count).astype(int)

    # Создание DataFrame с признаками для обучения
    x_test = df_test[['total_meters', 'floor', 'floors_count', 'rooms_count']].copy()
    x_test.loc[:, 'is_first'] = is_first
    x_test.loc[:, 'is_last'] = is_last

    # x_test = df_test[['total_meters', 'floor', 'floors_count', 'rooms_count']]
    # x_test = df_test[['floor', 'rooms_count', 'total_meters']]
    y_test = df_test['price']
    model = load(args.model)
    y_pred = model.predict(x_test)
    mae = mean_absolute_error(y_pred, y_test)
    logger.info(f'Test model {MODEL_SAVE_PATH} on {TEST_DATA}, MAE = {mae:.0f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model',
                        help='Model save path',
                        default=MODEL_SAVE_PATH)
    args = parser.parse_args()
    main(args)