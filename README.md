# Предиктивная аналитика больших данных

Учебный проект для демонстрации основных этапов жизненного цикла проекта предиктивной аналитики.  

## Installation 

Клонируйте репозиторий, создайте виртуальное окружение, активируйте и установите зависимости:  

```sh
git clone https://github.com/yourgit/pabd24
cd pabd24
python -m venv venv

source venv/bin/activate  # mac or linux
.\venv\Scripts\activate   # windows

pip install -r requirements.txt
```
# Структура проекта

```
pabd24
├── data
│   ├── proc
│   └── raw
├── log
├── models
├── notebooks
│   └── ipynb_checkpoints
│       └── EDA.ipynb
├── src
│   ├── downloaded_from_s3.py
│   ├── parse_cian.py
│   ├── predict_app.py
│   ├── preprocess_data.py
│   ├── test_model.py
│   ├── train_model.py
│   └── upload_to_s3.py
├── test
│   └── test_api.py
├── .env
├── .gitignore
├── LICENSE
└── README.md
```

Корневая директория проекта называется `pabd24`. Она содержит следующие поддиректории и файлы:

- `data` - директория для хранения данных, включая поддиректории `proc` и `raw`
- `log` - директория для логов
- `models` - директория для моделей
- `notebooks` - директория для ноутбуков, включая поддиректорию `ipynb_checkpoints` с файлом `EDA.ipynb`
- `src` - директория с исходными кодами, включая различные скрипты Python
- `test` - директория для тестов, включая `test_api.py`
- `.env` - файл с переменными окружения
- `.gitignore` - файл для игнорирования файлов в Git
- `LICENSE` - файл лицензии
- `README.md` - файл документации

Структура отображает иерархию папок и файлов в виде дерева с использованием символов `├──`, `└──` и `│` для визуализации вложенности.


## Usage

### 1. Сбор данных о ценах на недвижимость 

```markdown
Этот скрипт предназначен для сбора данных о ценах на недвижимость в Москве с использованием библиотеки `cianparser`. Скрипт извлекает данные о продаже квартир и сохраняет их в CSV файл.

## Установка

Для работы с проектом вам понадобится Python версии 3.8 или выше. 
Установите необходимые библиотеки прописав их в requirements.txt, либо через pip :

```bash
pip install flask
pip install flask-cors
pip install cianparser
pip install pandas

или в requirements.txt:
flask
flask-cors
cianparser
pandas
```

## Использование

Основной скрипт `parse_cian.py` собирает данные о продаже однокомнатных, двухкомнатных и трёхкомнатных квартир в Москве и сохраняет их в CSV файлы. Данные извлекаются с первых 50 страниц результатов поиска.

### Запуск скрипта

Чтобы запустить скрипт, выполните следующую команду:

```bash
python src\parse_cian.py
```

### Детали скрипта

Скрипт выполняет следующие шаги:

1. **Импорт необходимых библиотек**: `datetime` для генерации временной метки для имени CSV файла, `cianparser` для извлечения данных с Cian, `pandas` для работы с данными и сохранения их в CSV.

2. **Инициализация парсера**: Создание экземпляра `CianParser` для Москвы.

3. **Определение основной функции**: Генерация временной метки для имени файла, установка количества комнат в 1, определение пути для CSV файла, извлечение данных о продаже квартир с помощью `cianparser`, преобразование данных в DataFrame pandas, сохранение DataFrame в CSV файл.

### Дополнительные настройки

Метод `get_flats` в библиотеке `cianparser` поддерживает несколько дополнительных настроек:

- `deal_type`: Тип сделки (например, "sale" для продажи).
- `rooms`: Количество комнат (например, `(1,)` для однокомнатных квартир).
- `with_saving_csv`: Сохранение данных в CSV файл в процессе извлечения.
- `additional_settings`: Дополнительные параметры поиска, такие как начальная и конечная страницы поиска, тип объекта (например, "secondary" для вторичного рынка).

### Вывод

Скрипт сохраняет собранные данные в CSV файл в директории `data/raw`. Имя файла включает количество комнат и временную метку сбора данных.

### 2. Выгрузка данных в хранилище S3 
Для доступа к хранилищу скопируйте файл `.env` в корень проекта.  

```markdown
# Выгрузка данных в хранилище S3

Этот скрипт предназначен для выгрузки выбранных файлов в хранилище S3 с использованием библиотеки `boto3`. Скрипт загружает указанные CSV файлы в указанный бакет S3.

## Установка

Установите необходимые библиотеки с помощью pip или пропишите необходимые библиотеки в requirements.txt:

```bash
pip install boto3 python-dotenv

или в requirements.txt:
boto3
python-dotenv
```

## Настройка

Перед использованием скрипта, создайте файл `.env` в корневой директории проекта и добавьте в него ваши ключи доступа S3 хранилища:

```
KEY=your_aws_access_key_id
SECRET=your_aws_secret_access_key
```

## Использование

 `upload_to_s3.py` загружает указанные CSV файлы в бакет S3.

### Запуск скрипта

Чтобы запустить скрипт, выполните следующую команду:

```bash
python src\upload_to_s3.py
```

### Параметры скрипта

Скрипт принимает следующие параметры:

- `-i`, `--input`: Список локальных файлов данных для загрузки в хранилище S3. По умолчанию используется список файлов, указанный в переменной `CSV_PATH`.

### Пример кода

Вот полный код скрипта:

```python
"""Upload selected files to S3 storage"""
import argparse
from dotenv import dotenv_values
import boto3

BUCKET_NAME = 'pabd24'
YOUR_ID = '22'
CSV_PATH = ['data/raw/1_2024-05-13-00-25-22.csv',
            'data/raw/2_2024-05-13-00-12-15.csv',
            'data/raw/3_2024-05-13-00-29-59.csv']

config = dotenv_values(".env")

def main(args):
    client = boto3.client(
        's3',
        endpoint_url='https://storage.yandexcloud.net',
        aws_access_key_id=config['KEY'],
        aws_secret_access_key=config['SECRET']
    )

    for csv_path in args.input:
        object_name = f'{YOUR_ID}/' + csv_path.replace('\\', '/')
        client.upload_file(csv_path, BUCKET_NAME, object_name)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+',
                        help='Input local data files to upload to S3 storage',
                        default=CSV_PATH)
    args = parser.parse_args()
    main(args)
```

### Описание кода

1. **Импорт необходимых библиотек**: `argparse` для обработки аргументов командной строки, `dotenv_values` для загрузки переменных окружения из файла `.env`, `boto3` для взаимодействия с S3.

2. **Константы**:
    - `BUCKET_NAME`: Имя бакета S3.
    - `YOUR_ID`: Ваш уникальный идентификатор.
    - `CSV_PATH`: Список путей к CSV файлам, которые необходимо загрузить.

3. **Загрузка конфигурации**: Загрузка ключей доступа AWS из файла `.env`.

4. **Основная функция**:
    - Создание клиента S3 с использованием библиотеки `boto3`.
    - Загрузка каждого файла из списка `CSV_PATH` в указанный бакет S3.

5. **Запуск скрипта**: Обработка аргументов командной строки и вызов основной функции.


### 3. Загрузка данных из S3 на локальную машину  

```markdown
# Загрузка данных из S3 на локальную машину 

Этот скрипт предназначен для загрузки данных из хранилища S3 на локальную машину. Скрипт использует библиотеку `boto3` для взаимодействия с S3 и загружает указанные файлы на локальный диск.

## Установка

Установите необходимые библиотеки с помощью pip или пропишите необходимые библиотеки в requirements.txt:

```bash
pip install boto3 python-dotenv

или в requirements.txt:
boto3
python-dotenv
```

### Использование

Для запуска скрипта `downloaded_from_s3.py` выполните следующую команду:

```bash
python src/downloaded_from_s3.py -i <список_файлов>
```

#### Параметры

- `-i`, `--input`: Список локальных файлов данных для загрузки из хранилища S3. По умолчанию используется предопределенный список файлов:
    ```python
    CSV_PATH = ['data/raw/1_2024-05-13-00-25-22.csv',
                'data/raw/2_2024-05-13-00-12-15.csv',
                'data/raw/3_2024-05-13-00-29-59.csv']
    ```

### Пример

Для загрузки файлов по умолчанию выполните:

```bash
python src/downloaded_from_s3.py
```

Для загрузки конкретных файлов укажите их через параметр `-i`:

```bash
python src/downloaded_from_s3.py -i data/raw/1_2024-05-13-00-25-22.csv data/raw/2_2024-05-13-00-12-15.csv
```

### 4. Предварительная обработка данных  

# preprocess_data.py

Этот скрипт предназначен для предварительной обработки сырых данных и разделения их на тренировочный и валидационный наборы данных.

## Описание скрипта

Скрипт выполняет следующие шаги:

1. **Загрузка данных**:
    - Загружает несколько CSV файлов, указанных в списке `IN_FILES`.
    - Объединяет все загруженные файлы в один DataFrame.

2. **Обработка данных**:
    - Извлекает идентификатор URL и рассчитывает цену за квадратный метр.
    - Фильтрует данные, оставляя записи с ценой ниже 30 миллионов.
    - Формирует новый DataFrame с нужными колонками.

3. **Разделение данных**:
    - Разделяет данные на тренировочный и валидационный наборы в соответствии с заданным размером тренировочного набора (`TRAIN_SIZE`).

4. **Сохранение данных**:
    - Сохраняет тренировочный и валидационный наборы данных в файлы `train.csv` и `val.csv`.

## Использование

Запуск скрипта осуществляется из командной строки с возможностью указания входных файлов и размера тренировочного набора.

```bash
python preprocess_data.py -s 0.8 -i data/raw/1.csv data/raw/2.csv
```

Параметры:
- `-s`, `--split`: Размер тренировочного набора (по умолчанию 0.9).
- `-i`, `--input`: Список входных файлов (по умолчанию заданные в `IN_FILES`).

## Пример

Пример команды для запуска:

```bash
python preprocess_data.py -s 0.8 -i data/raw/1_2024-05-13-00-25-22.csv data/raw/2_2024-05-13-00-12-15.csv data/raw/3_2024-05-13-00-29-59.csv
```

Этот пример загрузит три CSV файла, объединит их, обработает данные и разделит их на тренировочный (80%) и валидационный (20%) наборы данных.

## Логирование

Скрипт ведет логирование своей работы в файл `log/preprocess_data.log`. Логи включают информацию о ходе выполнения и параметрах запуска.

## Структура папок

- `data/raw/` — исходные данные (CSV файлы).
- `data/proc/` — обработанные данные (train.csv и val.csv).
- `log/` — файлы логов.

## Требования

- Python 3.x
- Библиотеки: `pandas`, `argparse`, `logging`

## Код

```python
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
    main_dataframe = pd.read_csv(args.input[0], delimiter=';')
    for i in range(1, len(args.input)):
        data = pd.read_csv(args.input[i], delimiter=';')
        df = pd.DataFrame(data)
        main_dataframe = pd.concat([main_dataframe, df], axis=0)

    main_dataframe['url_id'] = main_dataframe['url'].map(lambda x: x.split('/')[-2])
    main_dataframe['price_per_meter'] = main_dataframe['price'] / main_dataframe['total_meters']

    new_dataframe = main_dataframe[['url_id', 'location', 'floor', 'rooms_count', 'total_meters', 'price', 'price_per_meter']].set_index('url_id')

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
```

### 5. Обучение модели 

todo Описание модели и входных параметров для предсказания здесь.  

### 6. Запуск приложения flask 

todo

### 7. Использование сервиса через веб интерфейс 

Для использования сервиса используйте файл `web/index.html`.  

## Лицензия

Проект распространяется под лицензией MIT. Смотрите файл [LICENSE](LICENSE) для деталей.
