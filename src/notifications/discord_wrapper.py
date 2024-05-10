import os
from dotenv import load_dotenv
import requests
import json
from requests_toolbelt.multipart.encoder import MultipartEncoder
import argparse

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


def send_discord_image(image_path):
    if os.environ.get("IS_LOCAL") == "yes":
        # No notifications on development
        return

    try:
        # Prepare the multipart data
        m = MultipartEncoder(
            fields={"file": (image_path, open(image_path, "rb"), "image/png")}
        )
        headers = {"Content-Type": m.content_type}
        webhook_url = os.environ.get("DISCORD_API_HOOK")

        # POST request to the webhook URL
        response = requests.post(webhook_url, data=m, headers=headers)

        # Check for success
        if response.status_code == 200:
            print("Image sent successfully.")
        else:
            print(f"Failed to send image. Status code: {response.status_code}")
    except Exception as e:
        print(e)


if __name__ == "__main__":
    # python src/notifications/discord_wrapper.py --message="Experiments done"
    # python src/notifications/discord_wrapper.py --image_path=artifacts/sweep_results_k.png
    parser = argparse.ArgumentParser(description="Send notifications to Discord.")
    parser.add_argument(
        "--message", type=str, help="The message to send as a notification."
    )
    parser.add_argument("--image_path", type=str, help="The path to the image to send.")

    args = parser.parse_args()

    if args.message is not None:
        send_discord_notification(args.message)

    if args.image_path is not None:
        send_discord_image(args.image_path)
