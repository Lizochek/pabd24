# # src/preprocess_data.py
# import argparse
# import logging
# import pandas as pd
#
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     filename='log/preprocess_data.log',
#     encoding='utf-8',
#     level=logging.DEBUG,
#     format='%(asctime)s %(message)s')
#
# IN_FILES = ['data/raw/1_2024-05-13-00-25-22.csv',
#             'data/raw/2_2024-05-13-00-12-15.csv',
#             'data/raw/3_2024-05-13-00-29-59.csv']
#
# OUT_TRAIN = 'data/proc/train.csv'
# OUT_VAL = 'data/proc/val.csv'
#
# TRAIN_SIZE = 0.9
#
# def main(args):
#     main_dataframe = pd.read_csv(args.input[0], delimiter=';')
#     for i in range(1, len(args.input)):
#         data = pd.read_csv(args.input[i], delimiter=';')
#         df = pd.DataFrame(data)
#         main_dataframe = pd.concat([main_dataframe, df], axis=0)
#
#     main_dataframe['url_id'] = main_dataframe['url'].map(lambda x: x.split('/')[-2])
#     main_dataframe['price_per_meter'] = main_dataframe['price'] / main_dataframe['total_meters']
#
#     # добавление новых фичей
#     # main_dataframe['building_age'] = 2024 - main_dataframe['year_built']
#     # main_dataframe['floor'] = main_dataframe['floor']
#     # main_dataframe['distance_to_metro'] = main_dataframe['distance_to_metro']
#     # main_dataframe['district'] = main_dataframe['district']
#
#     new_dataframe = main_dataframe[['url_id', 'location', 'floor', 'rooms_count' ,'total_meters','price', 'price_per_meter']].set_index('url_id')
#
#     new_df = new_dataframe[new_dataframe['price'] < 30_000_000]
#
#     border = int(args.split * len(new_df))
#     train_df, val_df = new_df[0:border], new_df[border:-1]
#     train_df.to_csv(OUT_TRAIN)
#     val_df.to_csv(OUT_VAL)
#     logger.info(f'Write {args.input} to train.csv and val.csv. Train set size: {args.split}')
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-s', '--split', type=float,
#                         help='Split test size',
#                         default=TRAIN_SIZE)
#     parser.add_argument('-i', '--input', nargs='+',
#                         help='List of input files',
#                         default=IN_FILES)
#     args = parser.parse_args()
#     main(args)
"""Transform raw data to train / val datasets """
import argparse
import logging
import pandas as pd

logger = logging.getLogger(__name__)
logging.basicConfig(
    filename='log/preprocess_data.log',
    encoding='utf-8',
    level=logging.DEBUG,
    format='%(asctime)s %(message)s')


IN_FILES = ['data/raw/1_2024-05-13-00-25-22.csv',
            'data/raw/2_2024-05-13-00-12-15.csv',
            'data/raw/3_2024-05-13-00-29-59.csv']

OUT_TRAIN = 'data/proc/train.csv'
OUT_VAL = 'data/proc/val.csv'

TRAIN_SIZE = 0.9


def main(args):
    main_dataframe = pd.read_csv(args.input[0], delimiter=',')
    for i in range(1, len(args.input)):
        data = pd.read_csv(args.input[i], delimiter=',')
        df = pd.DataFrame(data)
        main_dataframe = pd.concat([main_dataframe, df], axis=0)

    main_dataframe['url_id'] = main_dataframe['url'].map(lambda x: x.split('/')[-2])
    new_dataframe = main_dataframe[['url_id', 'location', 'floor', 'rooms_count' ,'total_meters','price']].set_index('url_id')

    new_df = new_dataframe[new_dataframe['price'] < 30_000_000]

    border = int(args.split * len(new_df))
    train_df, val_df = new_df[0:border], new_df[border:-1]
    train_df.to_csv(OUT_TRAIN)
    val_df.to_csv(OUT_VAL)
    logger.info(f'Write {args.input} to train.csv and val.csv. Train set size: {args.split}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=float,
                        help='Split test size',
                        default=TRAIN_SIZE)
    parser.add_argument('-i', '--input', nargs='+',
                        help='List of input files',
                        default=IN_FILES)
    args = parser.parse_args()
    main(args)

# """Transform raw data to train / val datasets """
# import argparse
# import logging
#
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import MinMaxScaler
# from sklearn.utils import shuffle
#
# logger = logging.getLogger(__name__)
# logging.basicConfig(
#     filename='log/preprocess_data.log',
#     encoding='utf-8',
#     level=logging.DEBUG,
#     format='%(asctime)s %(message)s')
#
#
# IN_FILES = ['data/raw/1_2024-05-13-00-25-22.csv',
#             'data/raw/2_2024-05-13-00-12-15.csv',
#             'data/raw/3_2024-05-13-00-29-59.csv']
#
# OUT_TRAIN = 'data/proc/train.csv'
# OUT_VAL = 'data/proc/val.csv'
#
# TRAIN_SIZE = 0.9
#
#
# def main(args):
#     main_dataframe = pd.read_csv(args.input[0], delimiter=',')
#     for i in range(1, len(args.input)):
#         data = pd.read_csv(args.input[i], delimiter=',')
#         df = pd.DataFrame(data)
#         main_dataframe = pd.concat([main_dataframe, df], axis=0)
#
#     # main_dataframe['url_id'] = main_dataframe['url'].map(lambda x: x.split('/')[-2])
#     # new_dataframe = main_dataframe[['url_id', 'total_meters', 'price']].set_index('url_id')
#     # Генерация новых фич
#     scaler = MinMaxScaler()
#     main_dataframe[['normalized_total_meters', 'normalized_price']] = scaler.fit_transform(
#         main_dataframe[['total_meters', 'price']])
#     # main_dataframe['log_total_meters'] = np.log1p(main_dataframe['total_meters'])
#     # main_dataframe['log_price'] = np.log1p(main_dataframe['price'])
#     main_dataframe['size_category'] = pd.cut(main_dataframe['total_meters'], bins=[0, 50, 100, 150, np.inf],
#                                              labels=['small', 'medium', 'large', 'extra_large'])
#     # main_dataframe['price_per_meter'] = main_dataframe['price'] / main_dataframe['total_meters']
#
#     new_dataframe = main_dataframe[
#         ['normalized_total_meters', 'normalized_price',  'size_category']]
#
#     new_df = new_dataframe[new_dataframe['price'] < 30_000_000]
#     df = shuffle(new_df)
#     border = int(args.split * len(new_df))
#     train_df, val_df = df[0:border], df[border:-1]
#     train_df.to_csv(OUT_TRAIN)
#     val_df.to_csv(OUT_VAL)
#     logger.info(f'Write {args.input} to train.csv and val.csv. Train set size: {args.split}')
#
#
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-s', '--split', type=float,
#                         help='Split test size',
#                         default=TRAIN_SIZE)
#     parser.add_argument('-i', '--input', nargs='+',
#                         help='List of input files',
#                         default=IN_FILES)
#     args = parser.parse_args()
#     main(args)
