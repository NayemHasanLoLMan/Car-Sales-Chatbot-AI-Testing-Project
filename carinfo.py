from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
import json
import os

# Initialize the FastAPI app
app = FastAPI()

# Add CORS middleware to allow requests from any origin
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow requests from any origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load car data from a JSON file
CAR_DATA_FILE = "car_info.json"
if not os.path.exists(CAR_DATA_FILE):
    raise FileNotFoundError(f"Car data file '{CAR_DATA_FILE}' not found.")

try:
    with open(CAR_DATA_FILE, "r") as f:
        car_info = json.load(f)
except json.JSONDecodeError:
    raise ValueError(f"Invalid JSON format in file: {CAR_DATA_FILE}")

# Initialize the Ollama LLaMA3 model
llm_model = OllamaLLM(model="llama3", temperature=0.3, max_tokens=512)

# Define the input data structure
class QuestionInput(BaseModel):
    question: str

# Set up Jinja2 templates for rendering HTML
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    """
    Serve the chatbot interface with an initial summary message.
    """
    initial_summary = (
        f"Welcome! I am your car assistant. Here's a quick overview:\n"
        f"- Make: {car_info.get('make', 'Unknown')}\n"
        f"- Model: {car_info.get('model', 'Unknown')}\n"
        f"- Type: {car_info.get('type', 'Unknown')}\n"
        f"- Performance: Top speed {car_info.get('performance', {}).get('top_speed', 'Unknown')}, "
        f"Acceleration {car_info.get('performance', {}).get('acceleration', 'Unknown')}.\n"
        "Feel free to ask me any specific questions."
    )
    return templates.TemplateResponse("chat.html", {"request": request, "summary": initial_summary})

@app.post("/ask")
async def ask_question(data: QuestionInput):
    """
    Process customer inquiries about specific vehicle details with focused,
    direct responses based on available information.
    """
    try:
        prompt = f"""
        You are a car assistant chatbot. Only use the following car information to answer questions:

        Car Information:
        {json.dumps(car_info, indent=4)}

        Response Rules:
        1. Only answer questions about the information provided above
        2. Keep responses short and direct - maximum 2 sentences
        3. If information isn't in the car data, say: "I apologize, this information is not available in my database."
        4. For pricing questions, respond: "For current pricing and offers, please visit our dealership."
        5. Don't redirect to other staff or departments
        6. Don't provide any additional information beyond what's asked
        7. For unclear questions, ask: "Could you please clarify your question about the {car_info['make']} {car_info['model']}?"
        8. Don't discuss or compare with other vehicles
        9. Don't speculate about features or specifications not listed
        10. Maintain a professional but straightforward tone

        Customer Question:
        "{data.question}"

        Response:
        """

        response = llm_model.generate([prompt])
        answer = response.generations[0][0].text.strip()

        return {"answer": answer}

    except Exception as e:
        raise HTTPException(
            status_code=500, 
            detail="I apologize, I'm unable to process your request right now."
        )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)



 
