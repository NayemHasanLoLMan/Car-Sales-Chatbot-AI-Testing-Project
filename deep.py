import time
import requests
import json

# Define API Key
OPENROUTER_API_KEY = "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"

# Define API URL and Headers
api_url = "https://openrouter.ai/api/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {OPENROUTER_API_KEY}",
    "Content-Type": "application/json",
}

# Define the conversation messages
messages = [
    {
        "role": "system",
        "content": "You are a friendly and helpful restaurant representative for 'Hasan's Restaurant'. "
                   "You will assist customers with inquiries about menu items, prices, and reservations. "
                   "Make up a simple lunch menu with 3-5 items and their prices, and respond professionally."
    },
    {
        "role": "user",
        "content": "Hi, What items do you have for lunch? Prices? How can I make a reservation?"
    }
]

# Retry logic for handling rate limits
max_retries = 5
retry_delay = 5  # Start with 5 seconds

for attempt in range(max_retries):
    response = requests.post(
        url=api_url,
        headers=headers,
        data=json.dumps({
            "model": "mistralai/mistral-7b-instruct",  # Use a supported model
            "messages": messages
        })
    )

    if response.status_code == 200:
        response_json = response.json()
        if "choices" in response_json and len(response_json["choices"]) > 0:
            print("\nAssistant Response:", response_json["choices"][0]["message"]["content"])
        else:
            print("\nError: No valid response received.")
        break  # Exit the loop if request was successful

    elif response.status_code == 429:
        print(f"Rate limit exceeded. Retrying in {retry_delay} seconds...")
        time.sleep(retry_delay)
        retry_delay *= 2  # Exponential backoff

    else:
        print(f"\nError: API request failed with status code {response.status_code}")
        print("Response:", response.text)
        break  # Exit loop on other errors
