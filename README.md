# 🚗💬 Car Sales Chatbot – AI Testing Project (OpenAI & DeepSeek)

This project is an experimental **AI-powered car sales chatbot** that tests different methods of collecting user preferences (both **static** and **dynamic**) to simulate real-world sales interactions and generate summaries. The system evaluates the capabilities of **OpenAI** and **DeepSeek** models for their conversational flow, reasoning, and data extraction performance.

The purpose of this project is to understand and improve the process of automated car sales assistance using state-of-the-art LLMs.

---

## 🎯 Project Goals

- 📋 Experiment with **static vs. dynamic user input collection**
- 🧠 Compare performance of **OpenAI vs. DeepSeek** in handling sales conversations
- 📝 Generate **summarized sales reports** after each session
- 🔍 Evaluate chatbot responsiveness and logical flow in handling multiple car-buying scenarios

---

## ✨ Features

- 🔄 **Static & Dynamic Data Capture**
  - Static: Predefined forms to collect user preferences
  - Dynamic: Real-time questioning and updates during chat

- 🤖 **Multimodel AI Engine**
  - Plug-and-play support for OpenAI and DeepSeek
  - Easy switching between backends for testing

- 🧾 **Conversation Summary Generator**
  - AI-generated session summary for sales agents
  - Includes budget, preferences, model, location, and more

- 🧪 **Purpose-built for R&D**
  - Designed to benchmark AI performance in real-world sales pipelines

---

## 🧠 AI Models Used

| Model Name | Provider   | Purpose                     |
|------------|------------|-----------------------------|
| GPT-4      | OpenAI     | Primary LLM for chat & logic|
| DeepSeek   | DeepSeek   | Model for experimental tests|

---




## 🚀 Getting Started

1. Clone the Repository
   
        git clone https://github.com/yourusername/car-sales-chatbot-testing.git
        cd car-sales-chatbot-testing

3. Setup Virtual Environment
   
        python -m venv env
        source env/bin/activate

4. Install Dependencies
   
       pip install -r requirements.txt

5. Configure API Keys

       cp .env.example .env




## 🛠️ Example Interaction

    Bot: Hello! What kind of car are you looking for today?
    User: I want something fuel efficient and under $20,000.
    Bot: Noted! Do you prefer a sedan or a hatchback?
    User: Sedan.
    Bot: Great! Here's a few options to get us started...



## 📊 Sales Summary Output


After each session, a summary like the following is generated:


    {
      "user_name": "John Doe",
      "budget": "$20,000",
      "preferences": ["fuel efficient", "sedan"],
      "suggested_models": ["Toyota Corolla", "Honda Civic"],
      "chat_duration": "7 minutes"
    }



## 🧪 Future Directions


 - Add lead generation form integration
 - Support voice-to-text interaction
 - Integrate real-time car inventory APIs
 - Evaluate performance using human test group




## 🔐 Environment Variables (.env)


    OPENAI_API_KEY=your_openai_key_here
    DEEPSEEK_API_KEY=your_deepseek_key_here


## 📄 License


MIT License – free to use and customize for your testing or internal tooling.

## 🙌 Contributing


Have ideas or want to benchmark another model? PRs are welcome! Let’s build smarter sales bots together.
