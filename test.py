import requests
from functools import lru_cache

# 🔹 Local database (Temporary storage for quick response)
local_database = {
    "What is the  capital of USA?": "Washington, D.C.",
    "Who is the president of USA?": "Joe Biden"
}

# 🔹 Gemini API call function (Optimized with timeout & response handling)
def api_call_to_gemini(question):
    api_url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-pro:generateContent"
    API_KEY = "API KEY"  # 👉 Replace with your actual API Key

    payload = {
        "contents": [{"role": "user", "parts": [{"text": question}]}]
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        # 🔹 API call with timeout (prevents hanging requests)
        response = requests.post(f"{api_url}?key={API_KEY}", json=payload, headers=headers, timeout=10)

        # 🔹 Process JSON response properly
        response_json = response.json()
        if "candidates" in response_json:
            return response_json["candidates"][0]["content"]["parts"][0]["text"].strip()
        else:
            print("❌ AI response does not contain a valid answer.")
            return None

    except requests.Timeout:
        print("❌ API Timeout! Try again later.")
        return None
    except Exception as e:
        print(f"❌ API call failed: {str(e)}")
        return None

# 🔹 Optimized get_answer function with caching
@lru_cache(maxsize=1000)  # Caching to avoid redundant API calls
def get_answer(question):
    if question in local_database:
        print("✅ Answer found in local database.")
        return local_database[question]  # Return answer from local storage
    else:
        print("❌ Answer not found in local database. Generating from AI model...")
        answer = api_call_to_gemini(question)  # Fetch answer from AI
        if answer:
            local_database[question] = answer  # Save new answer to local database
            print("✅ New answer saved to database.")
        return answer

# 🔹 User asks a question
user_question = input("Ask a question: ")  # User inputs a question
answer = get_answer(user_question)  # Process the question
print(f"🔹 Final Answer: {answer}")
