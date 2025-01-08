# aipsarg/tests/test_api.py
import unittest
from api.okx_api import OKXAPI
from api.mock_api import MockAPI

class TestAPI(unittest.TestCase):
   
    def test_okx_api_make_request(self):
        okx_api = OKXAPI()
        response = okx_api.make_request(method="GET",path="/api/v5/public/instruments",params = {"instType": "SPOT", "instId": "BTC-USDT"})
        self.assertIsNotNone(response)
    
    def test_mock_api_make_request(self):
        mock_api = MockAPI()
        response = mock_api.make_request(method="GET", path="/test")
        self.assertEqual(response["mocked"], True)

if __name__ == '__main__':
   unittest.main()
