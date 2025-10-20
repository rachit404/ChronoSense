# from groq import GroqClient
from utils.env import GROQ_API_KEY
import os
import requests

# Ensure your API key is set in the environment variables
api_key = GROQ_API_KEY
if not api_key:
    raise ValueError("GROQ_API_KEY environment variable is not set.")

# Define the API endpoint
url = "https://api.groq.com/openai/v1/models"

# Set up the headers
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

# Send the GET request to fetch available models
response = requests.get(url, headers=headers)

# Check if the request was successful
if response.status_code == 200:
    models = response.json()  # could be dict or list
    # Check the type of response
    if isinstance(models, dict) and "data" in models:
        models_list = models["data"]
    elif isinstance(models, list):
        models_list = models
    else:
        models_list = []
    
    if not models_list:
        print("No models returned by the API.")
    else:
        print("Available models:")
        # If each item is a string
        if isinstance(models_list[0], str):
            for model_name in models_list:
                print(f"- {model_name}")
        # If each item is a dict with 'id' key
        elif isinstance(models_list[0], dict) and "id" in models_list[0]:
            for model in models_list:
                print(f"- {model['id']}")
        else:
            print(models_list)
else:
    print(f"Failed to fetch models. Status code: {response.status_code}")
    print("Response:", response.text)
