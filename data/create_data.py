import pandas as pd
import json


#get measurements

import requests
url = "https://api.openaq.org/v2/measurements?location_id=8039&parameter=pm25&date_from=2023-05-31&date_to=2023-06-01&limit=1000&page=1&order_by=datetime"
headers = {"accept": "application/json"}
response = requests.get(url, headers=headers)
# print(response.text)
measurements_list = json.loads(response.text)
