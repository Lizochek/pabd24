# Предиктивная аналитика больших данных

Учебный проект для демонстрации основных этапов жизненного цикла проекта предиктивной аналитики.
Этот проект представляет собой веб-приложение на основе Flask для предсказания цены на недвижимость.
Приложение принимает параметры недвижимости через POST-запрос и возвращает предсказанную цену.

## Installation 

Клонируйте репозиторий, создайте виртуальное окружение, активируйте и установите зависимости:  

```sh
git clone https://github.com/Lizochek/pabd24
cd pabd24
python -m venv venv

source venv/bin/activate  # mac or linux
.\venv\Scripts\activate   # windows

pip install -r requirements.txt
```
## Структура проекта

```
pabd24
├── .dvc
│   ├── cache
│   ├── tmp
│   ├── .gitignore
│   └── config
├── data
│   ├── proc
│   ├── raw
│   └── raw.dvc
├── docs
│   └── report_3.md
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
│   ├── tupload_to_s3.py
│   └── utils.py
├── static
│   └── favicon.ico
├── test
│   ├── test_api.py
│   └── test_parallel.py
├── venv
├── web
│   └── index.html
├── .env
├── .gitignore
├── dvc.lock
├── dvc.yaml
├── LICENSE
├── requirements.txt
└── README.md
```

Корневая директория проекта называется `pabd24`. Она содержит следующие поддиректории и файлы:

- `.dvc` - директория, связанная с DVC (Data Version Control), содержит конфигурационные файлы и кеш
- `data` - директория для хранения данных
  - `proc` - директория для данных, разделённых на train, test, val
  - `raw` - директория для сырых данных
  - `raw.dvc` - файл DVC для управления версиями данных
- `docs` - директория для документации
  - `report_3.md` - пример файла документации
- `log` - директория для логов
- `models` - директория для моделей
- `notebooks` - директория для ноутбуков
  - `ipynb_checkpoints` - поддиректория для контрольных точек Jupyter Notebook
    - `EDA.ipynb` - пример ноутбука для разведочного анализа данных (EDA)
- `src` - директория с исходными кодами
  - `downloaded_from_s3.py` - скрипт для загрузки данных из S3
  - `parse_cian.py` - скрипт для парсинга данных с сайта Cian
  - `predict_app.py` - скрипт для запуска приложения Flask для предсказаний
  - `preprocess_data.py` - скрипт для предварительной обработки данных
  - `test_model.py` - скрипт для валидации модели
  - `train_model.py` - скрипт для обучения модели
  - `upload_to_s3.py` - скрипт для загрузки данных в S3
  - `utils.py` - утилиты, используемые в проекте
- `static` - директория для статических файлов
  - `favicon.ico` - иконка для веб-приложения
- `test` - директория для тестов
  - `test_api.py` - тесты для API
  - `test_parallel.py` - тесты для параллельного выполнения
- `venv` - виртуальное окружение Python
- `web` - директория для веб-ресурсов
  - `index.html` - HTML-файл для веб-интерфейса
- `.env` - файл с переменными окружения
- `.gitignore` - файл для игнорирования файлов в Git
- `dvc.lock` - файл блокировки DVC
- `dvc.yaml` - конфигурационный файл DVC
- `LICENSE` - файл лицензии
- `requirements.txt` - файл с зависимостями проекта
- `README.md` - файл документации

### Примечания

- `src` включает все скрипты, необходимые для работы проекта: от загрузки и обработки данных до обучения и валидации модели, а также утилиты.
- `test` содержит тесты для проверки работоспособности API и других компонентов.
- `dvc.lock` и `dvc.yaml` необходимы для управления версиями данных с помощью DVC.

## Usage

### 1. Сбор данных о ценах на недвижимость 
<li><strong><a href="https://github.com/Lizochek/pabd24/blob/main/src/parse_cian.py">parse_cian.py</a></strong> </li>

Этот скрипт предназначен для сбора данных о ценах на недвижимость в Москве с использованием библиотеки `cianparser`.
Скрипт `parse_cian.py` собирает данные о продаже однокомнатных, двухкомнатных и трёхкомнатных квартир в Москве и сохраняет их в CSV файлы. Данные извлекаются с первых 50 страниц результатов поиска.

#### Запуск скрипта

Чтобы запустить скрипт, выполните следующую команду:

```bash
python src\parse_cian.py
```
Параметры для парсинга можно изменить в скрипте.  
Подробности см. в [репозитории](https://github.com/Lizochek/pabd24)  

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
<li><strong><a href="https://github.com/Lizochek/pabd24/blob/main/src/upload_to_s3.py">upload_to_s3.py</a></strong> </li>
Для доступа к хранилищу скопируйте файл `.env` в корень проекта. 

Чтобы запустить скрипт, выполните следующую команду:

```bash
python src/upload_to_s3.py
```
Этот скрипт предназначен для выгрузки выбранных файлов в хранилище S3 с использованием библиотеки `boto3`. `upload_to_s3.py` загружает указанные CSV файлы в указанный бакет S3.

#### Настройка

Перед использованием скрипта, создайте файл `.env` в корневой директории проекта и добавьте в него ваши ключи доступа S3 хранилища:

```
KEY=your_aws_access_key_id
SECRET=your_aws_secret_access_key
```

##### Запуск скрипта

Чтобы запустить скрипт, выполните следующую команду:

```sh
python src/upload_to_s3.py -i data/raw/file.csv 
```  
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
<li><strong><a href="https://github.com/Lizochek/pabd24/blob/main/src/downloaded_from_s3.py">downloaded_from_s3.py</a></strong> </li>
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
<li><strong><a href="https://github.com/Lizochek/pabd24/blob/main/src/preprocess_data.py">preprocess_data.py</a></strong> </li>

Чтобы запустить скрипт, выполните следующую команду:

```bash
python src/preprocess_data.py
```

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
<li><strong><a href="https://github.com/Lizochek/pabd24/blob/main/src/train_model.py">train_model.py</a></strong> </li>

```bash
python src/train_model.py
```

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

### 6. Запуск приложения flask 
<li><strong><a href="https://github.com/Lizochek/pabd24/blob/main/src/predict_app.py">predict_app.py</a></strong> </li>

Данное приложение можно запустить на двух серверах: flask(порт 5000) и gunicorn(порт 8000), а также локально.

Для локального запуска приложения выполните следующую команду в терминале:

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
Введите площадь квартиры в поле area и нажмите submit.

#### Эндпоинт `/predict`

Для предсказания цены на недвижимость используйте эндпоинт `/predict`. Отправьте POST-запрос с параметрами недвижимости в формате JSON.


#### Объяснение работы кода

- Функция `predict` принимает параметры недвижимости, рассчитывает цену на основе площади и средней цены за квадратный метр и возвращает результат.
- Маршрут `/` отображает главную страницу с информационным сообщением.
- Маршрут `/favicon.ico` служит для отдачи иконки сайта.
- Маршрут `/predict` принимает POST-запрос с параметрами недвижимости в формате JSON, использует функцию `predict` для расчета цены и возвращает результат в формате JSON.


## Создание виртуальной машины и подключение по SSH. Клонирование проекта на виртуальную машину

* Создайте виртуальную машину на https://console.cloud.ru. Затем настройте группы безопасности для открытия портов 5000 и 8000.

* В Power Shell(в корне своего пользователя) создайте пары публичный-приватный ключ 
```shell
ssh-keygen
```

### Установка

* Подключитесь по SSH к виртуальной машине (в PowerShell в корне своего пользователя):

```sh
ssh user23@192.144.14.11
```

* Запустите оболочку bash:

```sh
bash
```

* Обновите индексы пакетов системы:

```sh
sudo apt update
```

* Установите модуль venv для Python 3:

```sh
sudo apt install python3-venv
```

* Клонируйте репозиторий, создайте виртуальное окружение, активируйте его и установите зависимости:

```sh
git clone https://github.com/Lizochek/pabd24
cd pabd24
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

* Открыть второе окно PowerShell в директории проекта `pabd24`, там где есть файл `.env`.  
Копирование файла `.env` из локальный машины на сервер выполняется командой:

```sh
scp .\.env user23@192.144.14.9:/home/user23/pabd24
```
#### Gunicorn сервер
* Установка и запуск: 
```shell
pip install gunicorn
gunicorn -b 0.0.0.0 -w 1 src.predict_app:app --daemon
```
* Для исследований следует опустить флажок `--daemon`, тогда можно легко остановить сервер нажатем `Ctr-C`. 
Остановить запущенный в фоне процесс можно командой 
```shell
pkill gunicorn
```
## Использование

### Загрузка данных из S3 на виртуальную машину

Создайте директорию для исходных данных и выполните скрипт для загрузки данных:

```sh
mkdir data/raw
python src/download_to_s3.py
```

### Предварительная обработка данных

Создайте директорию для обработанных данных и выполните скрипт для их предварительной обработки:

```sh
mkdir data/proc
python src/preprocess_data.py

cat log/preprocess_data.log # посмотреть лог
```

### Обучение и тестирование модели

Выполните команды для обучения и тестирования модели:

```sh
python src/train_model.py
python src/test_model.py
```
### Тестирование нагрузки

Выберите порт для тестрирования и укажите его в test/test_parallel.py и в index.html в endpoint.

* Подключитесь по SSH к виртуальной машине (в PowerShell в корне своего пользователя):

```sh
ssh user23@192.144.14.11
```
* Запустить оболочку bash
```shell
bash
```
* Создайте виртуальное окружение, активируйте его и установите зависимости:

```sh
git clone https://github.com/Lizochek/pabd24
cd pabd24
python3 -m venv venv
source venv/bin/activate
```
* Чтобы подтянуть изменения из удалённого репозитория выполните команду:
```sh
git pull
```
* Меняйте нагрузку в файле src/predict_app.py, если используете gunicorn вместо import utils напишите src.utils:
```sh
nano src/predict_app.py
```
* Запустите index.html в браузере, нажмите submit и проверьте предсказывается ли цена
* Зайдите в IDE и в терминале выполните команду:
```sh
python test\test_parallel.py
```
### 7. Использование сервиса через веб интерфейс 
<li><strong><a href="https://github.com/Lizochek/pabd24/blob/main/web/index.html">index.html</a></strong> </li>

Для использования сервиса используйте файл `web/index.html`.  

### 8. Запуск на сервере 

Порт, на котором приложение запущено в данный момент: 'http://192.144.14.11:8000/predict'
Ссылка на отчёт по исследованию поведения серверов flask и gunicorn под разными видами нагрузки [результат](https://github.com/Lizochek/pabd24/blob/main/docs/report_3.md)

### Лицензия
Проект распространяется под лицензией MIT. Смотрите файл [LICENSE](LICENSE) для деталей.

