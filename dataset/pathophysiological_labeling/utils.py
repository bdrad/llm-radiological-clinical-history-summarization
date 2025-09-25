import os
import json
from dotenv import load_dotenv
import requests
import time

load_dotenv("/mnt/sohn2022/Adrian/Utils/Credentials/.env")

API_KEY = os.environ.get('VERSA_API_KEY')  # Match the environment variable name to the name you used in the .env file
API_VERSION = os.environ.get('VERSA_API_VERSION')
RESOURCE_ENDPOINT = os.environ.get('RESOURCE_ENDPOINT')

RETRY_SECS = 15  
MAX_RETRIES = 5

API_DETAILS = {
    "gpt4o": {
        "DEPLOYMENT_ID": "gpt-4o-2024-05-13",
        "API_VERSION": "2024-10-01-preview"
    }
}

error_msg = "\nProvided your configuration parameters (API_KEY, API_VERSION, RESOURCE_ENDPOINT, deployment name) are valid, the majority of errors you may encounter with this code are attributable to temporary issues such as Azure server outages or other users who have triggered shared API rate limits for a given deployment. Please try again in a few minutes. However, if you receive a 401 Unauthorized access error, while your API key may have the correct length, most likely it is not a valid key for some other reason. In that event, please open a ticket with the Versa team at versa@ucsf.edu to review the key.\n"

def chat(prompt, model):
    DEPLOYMENT_ID = API_DETAILS[model]["DEPLOYMENT_ID"]
    API_VERSION = API_DETAILS[model]["API_VERSION"]
    url = f'{RESOURCE_ENDPOINT}/openai/deployments/{DEPLOYMENT_ID}/chat/completions?api-version={API_VERSION}'
    body = json.dumps({
        "messages": [{"role": "user", "content": prompt}]
    })
    headers = {'Content-Type': 'application/json', 'api-key': API_KEY}
    retries = 0
    while True:
        try:
            response = post_request(url, headers, body)
            output = json.loads(response.text).get('choices')[0].get('message').get('content')
            return output
            break
        except Exception as e:
            retries = exception_code(retries, DEPLOYMENT_ID, e)

def post_request(url, headers, body):
    response = requests.post(url, headers=headers, data=body)
    response.raise_for_status()
    return response

def exception_code(retries, deployment_id, e):
    if retries >= MAX_RETRIES:
        print(f'Failed attempt {retries+1} of {MAX_RETRIES+1}.')
        print(error_msg)
        
        assert False, f"Test failed for deployment: {deployment_id}, Error received: {e}"
    else:
        print(f'Failed attempt {retries+1} of {MAX_RETRIES + 1}. Waiting {RETRY_SECS} secs before next attempt...')
        
    retries += 1
    time.sleep(RETRY_SECS)
    return retries

def chat_gpt4o(prompt):
    return chat(prompt, "gpt4o")
