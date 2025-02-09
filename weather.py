import requests


api_address=("https://api.openweathermap.org/data/2.5/weather?q=Hyderabad&appid=565c19b515c6b7195e25d8f311bbfb68")
json_data=requests.get(api_address).json()

def temp():
    temprature=round(json_data["main"]["temp"]-273,1)
    return temprature


def des():
    description=json_data["weather"][0]["description"]
    return description
