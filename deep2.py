# from openai import OpenAI

# client = OpenAI(api_key="sk-7b8f607f94d04583927c5f7c9362ff2e", base_url="https://api.deepseek.com")

# response = client.chat.completions.create(
#     model="deepseek-chat",
#     messages=[
#         {"role": "system", "content": "You are a helpful assistant"},
#         {"role": "user", "content": "Hello"},
#     ],
#     stream=False
# )

# print(response.choices[0].message.content)





import requests

api_key = "sk-7b8f607f94d04583927c5f7c9362ff2e"
url = "https://api.deepseek.com/v1/chat/completions"

headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}

data = {
    "model": "deepseek-chat",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": "Hello"},
    ]
}

response = requests.post(url, headers=headers, json=data)
print(response.json())
