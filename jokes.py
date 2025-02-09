import requests


url="YOUR API KEY"
json_data=requests.get(url).json()

arr=["",""]
arr[0]=json_data["setup"]
arr[1]=json_data["punchline"]
def joke():
    return arr
