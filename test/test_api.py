import time
import unittest
import requests
from dotenv import dotenv_values

config = dotenv_values(".env")
ENDPOINT = 'http://192.144.14.11:5000'
HEADERS = {"Authorization": f"Bearer {config['APP_TOKEN']}"}


class TestApi(unittest.TestCase):
    def test_home(self):
        resp = requests.get(ENDPOINT)
        self.assertIn('Housing price service', resp.text)

    def test_api(self):
        data = {'area': 42}
        t0 = time.time()
        resp = requests.post(ENDPOINT +'/predict',
                             json=data,
                             headers=HEADERS)
        t1 = time.time()
        print(f'test predict: {t1 - t0}s')
        self.assertIn('price', resp.text)


if __name__ == '__main__':
    unittest.main()