import boto3
import cianparser
import pandas as pd
import datetime
import  boto3
from  dotenv import dotenv_values

config = dotenv_values('.env')
client = boto3.client(
    's3',
    endpoint_url='https://storage.yandex.net',
    aws_access_key_id = config['KEY'],
    aws_secret_access_key = config['SECRET']
)

moscow_parser = cianparser.CianParser(location="Москва")

def main():
    t = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    n_rooms = 2
    CSV_PATH = 'data/raw/{n_rooms}_{t}.csv'
    data = moscow_parser.get_flats(
        deal_type="sale",
        rooms=(n_rooms,),
        with_saving_csv=False,
        additional_settings={
            "start_page": 1,
            "end_page": 50,
            "object_type": "secondary"
        })
    df = pd.DataFrame(data)

    df.to_csv(CSV_PATH,
              encoding='utf-8',
              index = False)
    bucket_name = 'pabd24'
    objects_name ='22/' + CSV_PATH 
    client.upload_file(CSV_PATH, bucket_name, objects_name)


if __name__ == '__main__':
    main()
