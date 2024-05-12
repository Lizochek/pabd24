import boto3
import cianparser
import pandas as pd
import datetime
from dotenv import dotenv_values

# Загрузка конфигурации из файла .env
config = dotenv_values('.env')

# Инициализация клиента S3
client = boto3.client(
    's3',
    endpoint_url='https://storage.yandex.net',
    aws_access_key_id=config['KEY'],
    aws_secret_access_key=config['SECRET']
)

# Инициализация парсера для Москвы
moscow_parser = cianparser.CianParser(location="Москва")


def upload_to_s3(file_path, bucket, object_name):
    """
    Функция для загрузки файла в S3 хранилище
    """
    client.upload_file(file_path, bucket, object_name)


def send_to_s3():
    # Текущая дата и время для имени файла
    t = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    # Имя корзины
    bucket_name = 'pabd24'

    # Количество комнат и файлы для обработки
    room_counts = [1, 2, 3]  # Пример: квартиры с 1, 2 и 3 комнатами

    for n_rooms in room_counts:
        # Путь к CSV файлу
        csv_path = f'data/raw/{n_rooms}_{t}.csv'

        # Получение данных
        data = moscow_parser.get_flats(
            deal_type="sale",
            rooms=(n_rooms,),
            with_saving_csv=False,
            additional_settings={
                "start_page": 1,
                "end_page": 50,
                "object_type": "secondary"
            })

        # Создание DataFrame
        df = pd.DataFrame(data)

        # Сохранение DataFrame в CSV
        df.to_csv(csv_path, encoding='utf-8', index=False)

        # Имя объекта в S3
        object_name = f'22/{csv_path}'

        # Загрузка в S3
        upload_to_s3(csv_path, bucket_name, object_name)


if __name__ == '__main__':
    send_to_s3()
