import os
from dotenv import load_dotenv
import requests
import json

load_dotenv()


def send_discord_notification(message):
    if os.environ.get("IS_LOCAL") == "yes":
        # No notifications on development
        return

    try:
        data = {"content": message}
        headers = {"Content-Type": "application/json"}
        webhook_url = os.environ.get("DISCORD_API_HOOK")

        # POST request to the webhook URL
        response = requests.post(webhook_url, data=json.dumps(data), headers=headers)

        # Check for success
        if response.status_code == 204:
            print("Notification sent successfully.")
        else:
            print("Failed to send notification.")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    message = ":eggplant: Experiment completed!"
    send_discord_notification(message)
