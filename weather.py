import requests


api_address=("YOUR API KEY")
json_data=requests.get(api_address).json()

def temp():
    temprature=round(json_data["main"]["temp"]-273,1)
    return temprature


def des():
    description=json_data["weather"][0]["description"]
    return description
