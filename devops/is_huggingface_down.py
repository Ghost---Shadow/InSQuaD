import os
import requests
import dotenv

dotenv.load_dotenv()

HUGGINGFACE_API_KEY = os.environ.get("HUGGINGFACE_API_KEY")


def send_whoami_request():
    url = "https://huggingface.co/api/whoami-v2"
    headers = {
        "user-agent": "unknown/None; hf_hub/0.20.3; python/3.9.18",
        "Accept-Encoding": "gzip, deflate",
        "Accept": "*/*",
        "Connection": "keep-alive",
        "authorization": f"Bearer {HUGGINGFACE_API_KEY}",
    }

    response = requests.get(url, headers=headers)

    # Print the response status code and content for inspection
    print("Status Code:", response.status_code)
    print("Response:", response.json())


send_whoami_request()
