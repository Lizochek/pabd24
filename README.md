# Предиктивная аналитика больших данных

Учебный проект для демонстрации основных этапов жизненного цикла проекта предиктивной аналитики.
Этот проект представляет собой веб-приложение на основе Flask для предсказания цены на недвижимость.
Приложение принимает параметры недвижимости через POST-запрос и возвращает предсказанную цену.

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
## Структура проекта

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

## Usage

### 1. Сбор данных о ценах на недвижимость 


Этот скрипт предназначен для сбора данных о ценах на недвижимость в Москве с использованием библиотеки `cianparser`. Скрипт извлекает данные о продаже квартир и сохраняет их в CSV файл.

#### Использование

Основной скрипт `parse_cian.py` собирает данные о продаже однокомнатных, двухкомнатных и трёхкомнатных квартир в Москве и сохраняет их в CSV файлы. Данные извлекаются с первых 50 страниц результатов поиска.

#### Запуск скрипта

Чтобы запустить скрипт, выполните следующую команду:

```bash
python src\parse_cian.py
```

#### Детали скрипта

Скрипт выполняет следующие шаги:

1. **Импорт необходимых библиотек**: `datetime` для генерации временной метки для имени CSV файла, `cianparser` для извлечения данных с Cian, `pandas` для работы с данными и сохранения их в CSV.

2. **Инициализация парсера**: Создание экземпляра `CianParser` для Москвы.

3. **Определение основной функции**: Генерация временной метки для имени файла, установка количества комнат в 1, определение пути для CSV файла, извлечение данных о продаже квартир с помощью `cianparser`, преобразование данных в DataFrame pandas, сохранение DataFrame в CSV файл.

#### Дополнительные настройки

Метод `get_flats` в библиотеке `cianparser` поддерживает несколько дополнительных настроек:

- `deal_type`: Тип сделки (например, "sale" для продажи).
- `rooms`: Количество комнат (например, `(1,)` для однокомнатных квартир).
- `with_saving_csv`: Сохранение данных в CSV файл в процессе извлечения.
- `additional_settings`: Дополнительные параметры поиска, такие как начальная и конечная страницы поиска, тип объекта (например, "secondary" для вторичного рынка).

#### Вывод

Скрипт сохраняет собранные данные в CSV файл в директории `data/raw`. Имя файла включает количество комнат и временную метку сбора данных.

### 2. Выгрузка данных в хранилище S3 
Для доступа к хранилищу скопируйте файл `.env` в корень проекта.  

Этот скрипт предназначен для выгрузки выбранных файлов в хранилище S3 с использованием библиотеки `boto3`. Скрипт загружает указанные CSV файлы в указанный бакет S3.

#### Настройка

Перед использованием скрипта, создайте файл `.env` в корневой директории проекта и добавьте в него ваши ключи доступа S3 хранилища:

```
KEY=your_aws_access_key_id
SECRET=your_aws_secret_access_key
```

#### Использование

 `upload_to_s3.py` загружает указанные CSV файлы в бакет S3.

##### Запуск скрипта

Чтобы запустить скрипт, выполните следующую команду:

```bash
python src\upload_to_s3.py
```

#### Параметры скрипта

Скрипт принимает следующие параметры:

- `-i`, `--input`: Список локальных файлов данных для загрузки в хранилище S3. По умолчанию используется список файлов, указанный в переменной `CSV_PATH`.

#### Описание кода

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

Этот скрипт предназначен для загрузки данных из хранилища S3 на локальную машину. Скрипт использует библиотеку `boto3` для взаимодействия с S3 и загружает указанные файлы на локальный диск.

#### Использование

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

#### Пример

Для загрузки файлов по умолчанию выполните:

```bash
python src/downloaded_from_s3.py
```

Для загрузки конкретных файлов укажите их через параметр `-i`:

```bash
python src/downloaded_from_s3.py -i data/raw/1_2024-05-13-00-25-22.csv data/raw/2_2024-05-13-00-12-15.csv
```

### 4. Предварительная обработка данных  

#### preprocess_data.py

Этот скрипт предназначен для предварительной обработки сырых данных и разделения их на тренировочный и валидационный наборы данных.

#### Описание скрипта

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

#### Использование

Запуск скрипта осуществляется из командной строки с возможностью указания входных файлов и размера тренировочного набора.

```bash
python preprocess_data.py -s 0.8 -i data/raw/1.csv data/raw/2.csv
```

Параметры:
- `-s`, `--split`: Размер тренировочного набора (по умолчанию 0.9).
- `-i`, `--input`: Список входных файлов (по умолчанию заданные в `IN_FILES`).

#### Пример

Пример команды для запуска:

```bash
python preprocess_data.py -s 0.8 -i data/raw/1_2024-05-13-00-25-22.csv data/raw/2_2024-05-13-00-12-15.csv data/raw/3_2024-05-13-00-29-59.csv
```

Этот пример загрузит три CSV файла, объединит их, обработает данные и разделит их на тренировочный (80%) и валидационный (20%) наборы данных.

#### Логирование

Скрипт ведет логирование своей работы в файл `log/preprocess_data.log`. Логи включают информацию о ходе выполнения и параметрах запуска.

#### Структура папок

- `data/raw/` — исходные данные (CSV файлы).
- `data/proc/` — обработанные данные (train.csv и val.csv).
- `log/` — файлы логов.

### 5. Обучение модели 

#### train_model.py

Этот скрипт предназначен для обучения модели и сохранения контрольной точки модели.

#### Описание скрипта

Скрипт выполняет следующие шаги:

1. **Загрузка данных**:
    - Загружает тренировочные данные из CSV файла.

2. **Обучение модели**:
    - Использует алгоритм линейной регрессии для обучения модели на тренировочных данных.
    - Определяет зависимость цены от общей площади.

3. **Сохранение модели**:
    - Сохраняет обученную модель в файл, указанный пользователем, либо в файл по умолчанию.

4. **Логирование**:
    - Логирует информацию о процессе обучения и параметры модели.

#### Использование

Запуск скрипта осуществляется из командной строки с возможностью указания пути для сохранения модели.

```bash
python train_model.py -m models/my_model.joblib
```

Параметры:
- `-m`, `--model`: Путь для сохранения модели (по умолчанию `models/linear_regression_v01.joblib`).

#### Пример

Пример команды для запуска:

```bash
python train_model.py -m models/my_linear_model.joblib
```

Этот пример обучит модель линейной регрессии на тренировочных данных и сохранит её в файл `my_linear_model.joblib`.

#### Логирование

Скрипт ведет логирование своей работы в файл `log/train_model.log`. Логи включают информацию о ходе выполнения и параметры модели.

#### Структура папок

- `data/proc/` — обработанные данные (train.csv и val.csv).
- `models/` — сохраненные модели.
- `log/` — файлы логов.

### Обучение модели с использованием RandomForestRegressor

В случае использования модели случайного леса (RandomForestRegressor) скрипт выполняет дополнительные шаги:

- Загружает валидационные данные.
- Оценивает модель на валидационном наборе данных.
- Логирует  важность признаков.

#### Пример использования RandomForestRegressor:

```python
 df_train = pd.read_csv(TRAIN_DATA)
 x_train = df_train[['floor', 'rooms_count', 'total_meters']]
 y_train = df_train['price']

 model = RandomForestRegressor(n_estimators=100, random_state=42)
 model.fit(x_train, y_train)
 dump(model, args.model)
 logger.info(f'Saved to {args.model}')

 r2 = model.score(x_train, y_train)
 y_pred = model.predict(x_val)
```

### 6. Запуск приложения flask 

Для запуска Flask приложения выполните следующую команду:

```bash
python src/predict_app.py
```

Приложение будет доступно по адресу `http://0.0.0.0:5000`.

#### Использование

#### Главная страница

Перейдите по адресу `http://0.0.0.0:5000` для доступа к главной странице приложения. На главной странице будет отображено сообщение:

```
<h1>Housing price service.</h1> Use /predict endpoint
```

#### Эндпоинт `/predict`

Для предсказания цены на недвижимость используйте эндпоинт `/predict`. Отправьте POST-запрос с параметрами недвижимости в формате JSON.

#### Пример запроса

```bash
curl -X POST -H "Content-Type: application/json" -d '{"area": 50}' http://0.0.0.0:5000/predict
```

#### Пример ответа

```json
{
  "price": 10000000
}
```

#### Структура необходимая для работы скрипта

- `src/predict_app.py` — основной файл приложения Flask.
- `static/` — директория для статических файлов (например, `favicon.ico`).
- `requirements.txt` — список зависимостей проекта.

#### Объяснение работы кода

- Функция `predict` принимает параметры недвижимости, рассчитывает цену на основе площади и средней цены за квадратный метр и возвращает результат.
- Маршрут `/` отображает главную страницу с информационным сообщением.
- Маршрут `/favicon.ico` служит для отдачи иконки сайта.
- Маршрут `/predict` принимает POST-запрос с параметрами недвижимости в формате JSON, использует функцию `predict` для расчета цены и возвращает результат в формате JSON.


### 7. Использование сервиса через веб интерфейс 

Для использования сервиса используйте файл `web/index.html`.  

### Лицензия

Проект распространяется под лицензией MIT. Смотрите файл [LICENSE](LICENSE) для деталей.
