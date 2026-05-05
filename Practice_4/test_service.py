import requests


payload = {
    "age": 35,
    "sex": 1,
    "bmi": 27.5,
    "children": 1,
    "smoker": 0,
    "region": 2
}


response = requests.post(
    "http://127.0.0.1:8000/predict",
    json=payload,
    timeout=10
)

response.raise_for_status()

print("Request:", payload)
print("Response:", response.json())

assert "predicted_charges" in response.json()