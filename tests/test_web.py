import requests

resp = requests.post('http://localhost:5000/predict',
                     files={'file': open('classical.00002.wav', 'rb')})

print(resp.text)
