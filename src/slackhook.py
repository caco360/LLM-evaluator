import requests
import json
import dotenv
import os


webhook_url = os.getenv("webhook_url")

def send_slack_notification(text: str):
    slack_data = {'text': text}
    
    response = requests.post(
        webhook_url, 
        data=json.dumps(slack_data),
        headers={'Content-Type': 'application/json'}
    )

    if response.status_code != 200:
        raise ValueError(
            f'Request to Slack returned an error {response.status_code}, the response is:\n{response.text}'
        )
