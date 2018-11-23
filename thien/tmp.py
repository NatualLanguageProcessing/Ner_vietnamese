import requests

S = requests.Session()

URL = "https://en.wikipedia.org/w/api.php"

PARAMS = {
    'action':'query',
    'format':'json',
    'prop':'imageinfo',
    'titles':'yaw(rotation).jpg',
    'iiprop':'url'
}

R = S.get(url=URL, params=PARAMS)

DATA = R.json()
print(DATA)