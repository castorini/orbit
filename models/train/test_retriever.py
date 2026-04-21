import json
import requests

URL = "http://127.0.0.1:8280/retrieve"

queries = [
    "Nandan Thakur freshstack",
    "what is still fresh? a temporal rag benchmark",
    "Where did Freshstack won the honourable mention for best 2025 search benchmark award?",
    # "Name some Indian scholars associated with the Islamic University of Madinah.",
    # "Who is a Russian economist who served as senior policy advisor to a Russian president?",
]

payload = {"queries": queries, "topk": 5}

try:
    response = requests.post(URL, json=payload)
    response.raise_for_status()
    print(json.dumps(response.json(), indent=2))
except requests.exceptions.RequestException as e:
    print(f"Request failed: {e}")
