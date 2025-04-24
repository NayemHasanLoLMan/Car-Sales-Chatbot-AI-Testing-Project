from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_ollama import OllamaLLM
import asyncio
import json
import os
from typing import Dict, List, Optional, Tuple
from datetime import datetime

# Initialize FastAPI app
app = FastAPI()

# Middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files (CSS/JS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Templates directory for HTML files
templates = Jinja2Templates(directory="templates")

# Initialize LLM model with better temperature for more consistent responses
llm_model = OllamaLLM(model="llama3", temperature=0.3, max_tokens=512)


# Load car data from a JSON file
CAR_DATA_FILE = "car_info.json"
if not os.path.exists(CAR_DATA_FILE):
    raise FileNotFoundError(f"Car data file '{CAR_DATA_FILE}' not found.")

try:
    with open(CAR_DATA_FILE, "r") as f:
        car_info = json.load(f)
except json.JSONDecodeError:
    raise ValueError(f"Invalid JSON format in file: {CAR_DATA_FILE}")


# Define the input data structure
class QuestionInput(BaseModel):
    question: str


# Constants
DATA_DIR = "data"
CHAT_HISTORY_FILE = os.path.join(DATA_DIR, "chat_history.json")
USER_INFO_FILE = os.path.join(DATA_DIR, "user_info.json")
SESSION_FILE = os.path.join(DATA_DIR, "session.json")

# Ensure data directory exists
os.makedirs(DATA_DIR, exist_ok=True)

# Sample car inventory
inventory = [
    {"make": "Toyota", "type": "Sedan", "condition": "new", "price": 30000, "features": "Bluetooth, Backup Camera"},
    {"make": "Ford", "type": "SUV", "condition": "used", "price": 15000, "features": "Sunroof, Leather Seats"},
    {"make": "Honda", "type": "Coupe", "condition": "new", "price": 25000, "features": "Apple CarPlay, Heated Seats"},
    {"make": "BMW", "type": "Sedan", "condition": "new", "price": 45000, "features": "Navigation, Heated Seats"},
    {"make": "Mercedes-Benz", "type": "SUV", "condition": "used", "price": 38000, "features": "Panoramic Roof, Adaptive Cruise Control"},
    {"make": "Nissan", "type": "Truck", "condition": "new", "price": 35000, "features": "All-Wheel Drive, Backup Camera"},
    {"make": "Tesla", "type": "Sedan", "condition": "new", "price": 60000, "features": "Autopilot, Electric, Touchscreen Display"},
    {"make": "Chevrolet", "type": "SUV", "condition": "used", "price": 22000, "features": "Bluetooth, Parking Sensors"},
    {"make": "Hyundai", "type": "Hatchback", "condition": "new", "price": 20000, "features": "Android Auto, Lane Assist"},
    {"make": "Jeep", "type": "SUV", "condition": "new", "price": 32000, "features": "4x4, Off-Road Package"},
    {"make": "Audi", "type": "Coupe", "condition": "new", "price": 55000, "features": "Virtual Cockpit, Sport Package"},
    {"make": "Kia", "type": "Sedan", "condition": "used", "price": 18000, "features": "Apple CarPlay, Keyless Entry"},
    {"make": "Volkswagen", "type": "Hatchback", "condition": "new", "price": 24000, "features": "Heated Seats, Bluetooth"},
    {"make": "Subaru", "type": "SUV", "condition": "new", "price": 31000, "features": "All-Wheel Drive, Sunroof"},
    {"make": "Ram", "type": "Truck", "condition": "used", "price": 28000, "features": "Towing Package, Bed Liner"},
    {"make": "Lexus", "type": "SUV", "condition": "new", "price": 50000, "features": "Mark Levinson Audio, Heated Steering Wheel"},
    {"make": "Mazda", "type": "Sedan", "condition": "used", "price": 19000, "features": "Heads-Up Display, Leather Seats"},
    {"make": "Volvo", "type": "Wagon", "condition": "new", "price": 45000, "features": "Pilot Assist, Bowers & Wilkins Audio"},
    {"make": "Porsche", "type": "Convertible", "condition": "new", "price": 90000, "features": "Sport Exhaust, Adaptive Suspension"},
    {"make": "Chevrolet", "type": "Truck", "condition": "used", "price": 27000, "features": "Z71 Off-Road Package, Backup Camera"},
    {"make": "GMC", "type": "SUV", "condition": "new", "price": 40000, "features": "Adaptive Cruise Control, Heated and Cooled Seats"},
    {"make": "Ford", "type": "Truck", "condition": "new", "price": 45000, "features": "Pro Trailer Backup Assist, Blind Spot Monitoring"},
    {"make": "Dodge", "type": "Sedan", "condition": "used", "price": 21000, "features": "Remote Start, Bluetooth"},
    {"make": "Chrysler", "type": "Minivan", "condition": "new", "price": 36000, "features": "Stow 'n Go Seating, Entertainment System"},
    {"make": "Tesla", "type": "SUV", "condition": "new", "price": 85000, "features": "Electric, Autopilot, Long Range"},
    {"make": "Jaguar", "type": "SUV", "condition": "used", "price": 42000, "features": "Navigation, Meridian Sound System"},
    {"make": "Mini", "type": "Hatchback", "condition": "new", "price": 26000, "features": "Customizable Interior, Turbocharged"},
    {"make": "Toyota", "type": "Truck", "condition": "new", "price": 38000, "features": "TRD Package, Backup Camera"},
    {"make": "Nissan", "type": "Sedan", "condition": "used", "price": 16000, "features": "Keyless Entry, Blind Spot Warning"}
]

class LLMInput(BaseModel):
    input_text: str
    session_id: Optional[str] = None

class ConversationState:
    def __init__(self):
        self.current_stage: str = "initial"
        self.stages: List[str] = [
            "buy_lease",
            "car_type",
            "car_maker",
            "car_condition",
            "budget",
            "special_requirements"
        ]
        self.user_info: Dict = {
            "buy_lease": None,
            "car_type": None,
            "car_maker": None,
            "car_condition": None,
            "budget": None,
            "special_requirements": None
        }
        self.conversation_history: List[Dict] = []
        self.session_id: str = datetime.now().strftime("%Y%m%d_%H%M%S")

    def get_next_stage(self) -> Optional[str]:
        """Get the next empty stage that needs to be filled."""
        for stage in self.stages:
            if self.user_info[stage] is None:
                return stage
        return None

    def reset(self):
        """Reset the conversation state."""
        self.current_stage = "initial"
        self.user_info = {key: None for key in self.user_info}
        self.conversation_history = []

def save_state(state: ConversationState):
    """Save conversation state to file."""
    try:
        state_data = {
            "current_stage": state.current_stage,
            "user_info": state.user_info,
            "conversation_history": state.conversation_history,
            "session_id": state.session_id
        }
        with open(os.path.join(DATA_DIR, f"session_{state.session_id}.json"), "w") as f:
            json.dump(state_data, f, indent=4)
    except Exception as e:
        print(f"Error saving state: {e}")

def load_state(session_id: str) -> Optional[ConversationState]:
    """Load conversation state from file."""
    try:
        file_path = os.path.join(DATA_DIR, f"session_{session_id}.json")
        if os.path.exists(file_path):
            with open(file_path, "r") as f:
                data = json.load(f)
            state = ConversationState()
            state.current_stage = data["current_stage"]
            state.user_info = data["user_info"]
            state.conversation_history = data["conversation_history"]
            state.session_id = data["session_id"]
            return state
        return None
    except Exception as e:
        print(f"Error loading state: {e}")
        return None

def check_relevance(user_input: str, current_stage: str) -> Tuple[bool, str]:
    """Check if the user input is relevant to the current conversation stage."""
    prompt = f"""
    Analyze if the following user input is relevant to collecting {current_stage} information for a car purchase/lease:
    
    User input: "{user_input}"
    Current stage: "{current_stage}"
    
    Return your response in this exact format:
    RELEVANT: [true/false]
    REASON: [brief explanation]
    """
    
    try:
        response = llm_model.generate([prompt])
        result = response.generations[0][0].text.strip()
        
        relevant = "RELEVANT: true" in result.lower()
        reason = result.split("REASON:")[1].strip() if "REASON:" in result else "No explanation provided"
        
        return relevant, reason
    except Exception as e:
        print(f"Error in relevance checking: {e}")
        return True, "Relevance check failed, proceeding with input processing"

def extract_key_info(user_input: str) -> Dict:
    """Extract key information from user input using LLM with robust JSON handling."""
    prompt = f"""
    You are a car sales assistant. Analyze this message and return ONLY a JSON object:

    Message: "{user_input}"

    Return ONLY a JSON object in this EXACT format with no other text:
    {{
        "buy_lease": null,
        "car_type": null,
        "car_maker": null,
        "car_condition": null,
        "budget": null,
        "special_requirements": null
    }}

    Rules:
    1. For "buy_lease": Set to "buy" or "lease" if explicitly stated, else null
    2. For "car_type": Set to vehicle type (sedan/suv/truck/etc) if mentioned, else null
    3. For "car_maker": Set to car brand if mentioned, else null
    4. For "car_condition": Set to "new" or "used" if mentioned, else null
    5. For "budget": Set to numeric value if mentioned (handle "30k" or "30,000"), else null
    6. For "special_requirements": Set to a comma-separated string of features (e.g., "GPS, sunroof, leather seats"), else null

    IMPORTANT: Return ONLY the JSON object with no other text or explanation.
    """
    
    try:
        response = llm_model.generate([prompt])
        result = response.generations[0][0].text.strip()
        
        # Clean up the response to ensure valid JSON
        json_start = result.find("{")
        json_end = result.rfind("}") + 1
        if json_start == -1 or json_end == -1:
            raise ValueError("No valid JSON found in response")
            
        json_str = result[json_start:json_end]
        json_str = (json_str
                   .replace('None', 'null')
                   .replace('True', 'true')
                   .replace('False', 'false')
                   .replace("'", '"'))
        
        extracted_info = json.loads(json_str)
        
        # Normalize the extracted information
        normalized_info = {
            "buy_lease": None,
            "car_type": None,
            "car_maker": None,
            "car_condition": None,
            "budget": None,
            "special_requirements": None
        }
        
        for key in normalized_info:
            value = extracted_info.get(key)
            if value and str(value).lower() not in ['null', 'none', '']:
                if key == "budget":
                    normalized_info[key] = parse_budget(value)
                elif key == "special_requirements":
                    # Handle special requirements as a comma-separated string
                    if isinstance(value, dict):
                        # Convert dict to comma-separated string
                        features = [k for k, v in value.items() if v]
                        normalized_info[key] = ", ".join(features) if features else None
                    elif isinstance(value, str):
                        normalized_info[key] = value
                    else:
                        normalized_info[key] = str(value)
                else:
                    normalized_info[key] = str(value).lower()
        
        return normalized_info
    
    except Exception as e:
        print(f"Error in extract_key_info: {str(e)}")
        return {k: None for k in ["buy_lease", "car_type", "car_maker", "car_condition", "budget", "special_requirements"]}


def check_relevance(user_input: str, current_stage: str) -> Tuple[bool, str]:
    """Check if the user input is relevant to the current conversation stage with improved context awareness."""
    stage_context = {
        "buy_lease": "whether the user wants to buy or lease a car",
        "car_type": "what type of vehicle (sedan, SUV, etc.) the user is interested in",
        "car_maker": "which car brand or manufacturer the user prefers",
        "car_condition": "whether the user wants a new or used car",
        "budget": "how much the user is willing to spend",
        "special_requirements": "any specific features or requirements the user has"
    }

    prompt = f"""
    You are a car sales assistant having a conversation with a customer.
    
    Current topic: {stage_context.get(current_stage, "initial car buying discussion")}
    User message: "{user_input}"

    Task: Determine if this message contains information relevant to the current topic.

    Rules:
    1. Consider both direct and indirect references
    2. Account for conversational context
    3. Look for implicit information
    4. Consider alternative phrasings
    
    Response format:
    RELEVANT: [true/false]
    REASON: [brief explanation of why the input is relevant or not]
    EXTRACTED_INFO: [what specific information was found, if any]
    """
    
    try:
        response = llm_model.generate([prompt])
        result = response.generations[0][0].text.strip()
        
        # Parse the structured response
        lines = result.split('\n')
        relevant = any(line.lower().startswith('relevant: true') for line in lines)
        reason = next((line.split('REASON:')[1].strip() 
                      for line in lines if line.startswith('REASON:')), 
                     "No explanation provided")
        
        return relevant, reason
    except Exception as e:
        print(f"Error in relevance checking: {e}")
        return True, "Proceeding with input processing due to error in relevance check"

def parse_budget(budget_str: str) -> Optional[int]:
    """Parse budget string to integer with improved handling."""
    try:
        if isinstance(budget_str, (int, float)):
            return int(budget_str)
        
        # Remove common currency symbols and separators
        cleaned = (
            str(budget_str)
            .replace('$', '')
            .replace(',', '')
            .replace('usd', '')
            .replace('around', '')
            .strip()
            .lower()
        )
        
        # Handle "k" notation (e.g., "30k" = 30000)
        if 'k' in cleaned:
            cleaned = cleaned.replace('k', '')
            multiplier = 1000
        else:
            multiplier = 1
        
        # Convert to float first to handle decimal values
        value = float(cleaned)
        
        # Convert to integer and apply multiplier
        return int(value * multiplier)
    
    except (ValueError, AttributeError, TypeError) as e:
        print(f"Error parsing budget: {e}")
        return None

def format_car_recommendations(cars: List[Dict]) -> str:
    """Format car recommendations in a user-friendly way."""
    if not cars:
        return "I couldn't find any exact matches for your preferences. Would you like to adjust your criteria?"
    
    response = "Based on your preferences, here are some recommendations:\n\n"
    for i, car in enumerate(cars, 1):
        response += f"{i}. {car['make']} {car['type']}\n"
        response += f"   Price: ${car['price']:,}\n"
        response += f"   Condition: {car['condition']}\n"
        response += f"   Features: {car['features']}\n\n"
    
    response += "Would you like more details about any of these vehicles?"
    return response

def search_inventory(user_info: Dict) -> List[Dict]:
    """Search inventory based on user preferences."""
    filtered_cars = []
    for car in inventory:
        if all([
            user_info["car_type"] is None or user_info["car_type"].lower() == car["type"].lower(),
            user_info["car_maker"] is None or user_info["car_maker"].lower() in car["make"].lower(),
            user_info["car_condition"] is None or user_info["car_condition"] == car["condition"],
            user_info["budget"] is None or car["price"] <= user_info["budget"]
        ]):
            filtered_cars.append(car)
    
    return filtered_cars[:5] # Limit to top 5 matches

def generate_response(state: ConversationState, user_input: str) -> str:
    """Generate contextually appropriate responses based on conversation state."""
    next_stage = state.get_next_stage()
    
    if not next_stage:
        cars = search_inventory(state.user_info)
        return format_car_recommendations(cars)
    
    # Create context-aware prompt based on conversation state
    collected_info = [f"{k}: {v}" for k, v in state.user_info.items() if v is not None]
    collected_info_str = "\n".join(collected_info) if collected_info else "No information collected yet"
    
    prompt = f"""
    You are a helpful car sales assistant. Generate a natural response based on this context:

    Previous user message: "{user_input}"
    Information collected so far:
    {collected_info_str}
    
    Next information needed: {next_stage.replace('_', ' ')}

    Requirements for your response:
    1. Acknowledge any relevant information the user just provided
    2. Ask about {next_stage.replace('_', ' ')} naturally
    3. Keep the response conversational and brief (max 2 sentences)
    4. Don't mention "next stage" or use technical terms
    
    Example good responses:
    - "I see you're interested in SUVs. What's your preferred budget range?"
    - "A new car sounds great. Are there any specific brands you're interested in?"
    """
    
    try:
        response = llm_model.generate([prompt])
        return response.generations[0][0].text.strip()
    except Exception as e:
        print(f"Error generating response: {e}")
        return f"What's your preference regarding {next_stage.replace('_', ' ')}?"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Serve the chat interface."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/")
async def process_with_llm(data: LLMInput):
    try:
        # Step 1: Load or create session state
        if data.session_id:
            state = load_state(data.session_id)
            if not state:
                state = ConversationState()
                state.session_id = data.session_id
        else:
            state = ConversationState()

        print(f"Session ID: {state.session_id}")
        print(f"Received input: {data.input_text}")

        # Step 2: Extract information from input
        extracted_info = extract_key_info(data.input_text)
        print(f"Extracted info: {extracted_info}")

        # Step 3: Update conversation state with new information
        for key, value in extracted_info.items():
            if value is not None:
                state.user_info[key] = value
        print(f"Updated state: {state.user_info}")

        # Step 4: Check if all required information is collected
        all_info_collected = all([
            state.user_info["buy_lease"],
            state.user_info["car_type"],
            state.user_info["car_maker"],
            state.user_info["car_condition"],
            state.user_info["budget"],
            state.user_info["special_requirements"]
        ])

        # Step 5: Generate appropriate response or summary
        if all_info_collected:
            # Generate a summary of requirements
            summary = generate_requirements_summary(state.user_info)
            response = summary
        else:
            # Generate next question based on missing information
            response = generate_response(state, data.input_text)

        # Step 6: Save updated conversation state
        state.conversation_history.append({
            "user": data.input_text,
            "assistant": response,
            "timestamp": datetime.now().isoformat()
        })
        save_state(state)

        # Step 7: Return response with current state and session ID
        return {
            "response": response,
            "session_id": state.session_id,
            "state": state.user_info,
            "completed": all_info_collected
        }

    except Exception as e:
        print(f"Error processing input: {str(e)}")
        return {
            "response": "I apologize, but I'm having trouble understanding. Could you please rephrase that?",
            "session_id": data.session_id or "unknown",
            "state": None,
            "completed": False
        }

def generate_requirements_summary(user_info: Dict) -> str:
    """Generate a natural language summary of the user's requirements."""
    # Format budget as currency
    budget = f"${user_info['budget']:,}" if user_info['budget'] else "Not specified"
    
    # Create the summary
    summary = "Great! I've gathered all your requirements. Here's what you're looking for:\n\n"
    
    # Format the basic requirements
    summary += f"• You want to {user_info['buy_lease']} a {user_info['car_condition']} {user_info['car_maker']} {user_info['car_type']}\n"
    summary += f"• Your budget is {budget}\n"
    
    # Format special requirements in a cleaner way
    if user_info['special_requirements']:
        features = user_info['special_requirements'].strip()
        summary += f"• Special requirements: {features}\n"
    
    summary += "\nBased on your requirements, I will look for the available cars."
    
    return summary


def search_partial_inventory(user_info: Dict) -> List[Dict]:
    """
    Search inventory for partial matches when exact matches are not available.
    Matches on at least one user-provided criterion.
    """
    partial_matches = []

    for car in inventory:
        match_score = 0

        # Assign weights to each matching field
        if user_info["car_type"] and user_info["car_type"].lower() == car["type"].lower():
            match_score += 3
        if user_info["car_maker"] and user_info["car_maker"].lower() in car["make"].lower():
            match_score += 2
        if user_info["car_condition"] and user_info["car_condition"] == car["condition"]:
            match_score += 2
        if user_info["budget"] and car["price"] <= user_info["budget"]:
            match_score += 1

        if match_score > 0:  # Add cars with non-zero match score
            partial_matches.append((car, match_score))

    # Sort by match score in descending order and return the top 5 matches
    partial_matches.sort(key=lambda x: x[1], reverse=True)
    return [car for car, score in partial_matches[:5]]


def format_partial_recommendations(cars: List[Dict]) -> str:
    """Format partial car recommendations in a user-friendly way."""
    if not cars:
        return "I couldn't find any vehicles matching your preferences. Would you like to adjust your criteria?"

    response = "I couldn't find an exact match, but here are some vehicles that are close to your preferences:\n\n"
    for i, car in enumerate(cars, 1):
        response += f"{i}. {car['make']} {car['type']}\n"
        response += f"   Price: ${car['price']:,}\n"
        response += f"   Condition: {car['condition']}\n"
        response += f"   Features: {car['features']}\n\n"

    response += "Would you like more details about any of these vehicles?"
    return response




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
        3. If information isn't in the car data, say: "I apologize, I don't have that information avalable right now."
        4. For pricing questions, respond: "For current pricing and offers, it  can vary based on location and dealership.therefore, I recommend checking with your local dealership."
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
