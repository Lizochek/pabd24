import time
from multiprocessing import Pool
import requests
from dotenv import dotenv_values

proxies = {
    "http:" "",
    "https:" "",
}

config = dotenv_values(".env")
endpoint = 'http://192.144.14.11:5000/predict'
HEADERS = {"Authorization": f"Bearer {config['APP_TOKEN']}"}


def do_request(area: int) -> str:
    data = {'total_meters': area}
    t0 = time.time()
    try:
        resp = requests.post(
            endpoint,
            json=data,
            headers=HEADERS
        )
        resp.raise_for_status()  # Проверяем, что запрос успешен
        response_text = resp.text
    except requests.exceptions.RequestException as e:
        response_text = f"Error: {e}"
    t = time.time() - t0
    return f'Waited {t:0.2f} sec {response_text}'


def test_10():
    with Pool(10) as p:
        results = p.map(do_request, range(10, 110, 10))
        for result in results:
            print(result)


if __name__ == '__main__':
    test_10()
