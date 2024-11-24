import os
from dotenv import load_dotenv
import requests
import json
from requests_toolbelt.multipart.encoder import MultipartEncoder
import argparse

load_dotenv()


def send_discord_notification_helper(message):
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
            print(f"Failed to send notification. {response.json()}")
    except Exception as e:
        print(f"Exception during notification: {e}")


def send_discord_notification(message):
    max_length = 1994  # Subtract 6 to account for the addition of code block markers
    chunks = []
    in_code_block = message.startswith("```")

    while message:
        if in_code_block:
            # Find the next code block end or adjust for the maximum message length
            next_code_block_end = message.find("```", 3)
            if next_code_block_end != -1 and next_code_block_end + 3 <= max_length:
                split_point = next_code_block_end + 3
            else:
                split_point = max_length
        else:
            split_point = min(max_length, len(message))

        # Ensure the split happens at the last newline within the allowed length
        last_newline = message.rfind("\n", 0, split_point)
        if last_newline != -1:
            split_point = last_newline + 1

        chunks.append(message[:split_point])
        message = message[split_point:]
        if in_code_block and message.startswith("```"):
            in_code_block = False
        elif not in_code_block and message.startswith("```"):
            in_code_block = True

    # Process each chunk for sending
    for chunk in chunks:
        if chunk.count("```") % 2 != 0:
            # Balance unclosed code blocks for each chunk
            chunk += "```"
        send_discord_notification_helper(chunk)


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
