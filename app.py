# from fastapi import FastAPI, Request, HTTPException
# from fastapi.responses import HTMLResponse, JSONResponse
# from starlette.middleware.sessions import SessionMiddleware
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from langchain_ollama import OllamaLLM
# from starlette.responses import FileResponse
# import logging
# from fastapi.middleware.cors import CORSMiddleware
# import json
# from datetime import datetime
# from pathlib import Path
# import asyncio

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[
#         logging.FileHandler('app.log'),
#         logging.StreamHandler()
#     ]
# )
# logger = logging.getLogger("automotive_sales")

# # Initialize FastAPI app
# app = FastAPI(title="Automotive Sales Assistant")

# # Middleware configuration
# app.add_middleware(
#     SessionMiddleware, 
#     secret_key="your-secret-key-here",
#     max_age=3600
# )
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Static files and templates
# app.mount("/static", StaticFiles(directory="static"), name="static")
# templates = Jinja2Templates(directory="templates")

# # Ensure data directory exists
# DATA_DIR = Path("data")
# DATA_DIR.mkdir(exist_ok=True)

# # Improved system context tailored for the updated flow
# SYSTEM_CONTEXT = """
# Task:
# Assist customers in buying or leasing a car by guiding them through the process of selecting a vehicle, exploring budget-friendly options, and providing detailed information about available models and deals.

# Context:
# Car AI is an intelligent assistant for a car dealership. It helps customers make informed decisions about buying or leasing vehicles. The chatbot provides recommendations based on customer preferences, budget, and the latest deals while ensuring a smooth and engaging experience.

# Example:
# Customer: "Hi, I’m looking for a car."
# Car AI: "Welcome to Car AI! Are you looking to buy or lease a car today?"
# Customer: "I want to lease a car."
# Car AI: "Great! What type of car are you interested in—Sedan, SUV, Truck, or Van?"
# Customer: "An SUV."
# Car AI: "What’s your budget range? For example, $30K, $40K, or higher?"
# Customer: "$30K."
# Car AI: "Based on your preference and budget, here are three SUVs you might like:
# 1. Toyota RAV4 LE (Lease for $299/month).
# 2. Honda CR-V EX (Lease for $319/month).
# 3. Hyundai Tucson SE (Lease for $289/month).
# Would you like more details about any of these options?"

# Persona:
# Car AI is professional, knowledgeable, and approachable. It communicates clearly and concisely, with a tone that is friendly, informative, and customer-centric.

# Format:
# 1. Greeting: Start with a warm welcome and introduce Car AI.
# 2. Step-by-Step Process: Follow a structured, interactive approach to gather preferences and suggest cars.
# 3. Options: Present clear, concise recommendations.
# 4. Details: Provide in-depth information upon request.
# 5. Deals: Share available offers and incentives.
# 6. Closing: Offer further assistance and thank the customer.

# Tone:
# The tone should be:
# - Friendly: Make the customer feel valued and understood.
# - Informative: Share relevant details and options with clarity.
# - Professional: Maintain a polished, respectful tone throughout the conversation.
# - Engaging: Keep the interaction lively and tailored to the customer’s needs.

# Step-by-Step Process:
# Step 1: "Welcome to Car AI! Are you looking to buy or lease a car today?"
# Step 2: "Great! What type of car are you interested in—Sedan, SUV, Truck, or Van?"
# Step 3: "What’s your budget range? For example, $30K, $40K, or higher?"
# Step 4: "Based on your preference and budget, here are three options:
# 1. [Car Model 1]
# 2. [Car Model 2]
# 3. [Car Model 3]
# Would you like to know more about any of these models?"
# Step 5: If the customer selects a model:
# "Here are the details for [Car Model]: [Detailed information]."
# Step 6: "Would you like to explore any current deals or incentives for the selected car(s)?"
# Step 7: If the customer agrees:
# "Here are the latest deals for [Car Model]: [Deals and offers]."
# Closing: "Thank you for exploring with Car AI! Let me know if you have any other questions or need further assistance. Have a great day!"
# """

# # Initialize LLaMA model
# model = OllamaLLM(
#     model="llama3",
#     temperature=0.7,
#     top_p=0.95,
#     max_tokens=1024
# )

# class CustomerDataManager:
#     def __init__(self, data_dir: Path):
#         self.data_dir = data_dir
#         self.current_interactions_file = data_dir / "current_interactions.json"
#         self.completed_interactions_file = data_dir / "completed_interactions.json"
#         self._initialize_files()

#     def _initialize_files(self):
#         for file in [self.current_interactions_file, self.completed_interactions_file]:
#             if not file.exists():
#                 file.write_text('{}')

#     def save_interaction(self, session_id: str, user_info: dict, conversation: list):
#         current_data = self._load_json(self.current_interactions_file)
#         current_data[session_id] = {
#             "user_info": user_info,
#             "conversation": conversation,
#             "last_updated": datetime.now().isoformat()
#         }
#         self._save_json(self.current_interactions_file, current_data)

#     def complete_interaction(self, session_id: str):
#         current_data = self._load_json(self.current_interactions_file)
#         if session_id in current_data:
#             completed_data = self._load_json(self.completed_interactions_file)
#             completed_data[session_id] = current_data[session_id]
#             completed_data[session_id]["completed_at"] = datetime.now().isoformat()
#             self._save_json(self.completed_interactions_file, completed_data)
#             del current_data[session_id]
#             self._save_json(self.current_interactions_file, current_data)

#     def _load_json(self, file_path: Path) -> dict:
#         try:
#             return json.loads(file_path.read_text())
#         except json.JSONDecodeError:
#             return {}

#     def _save_json(self, file_path: Path, data: dict):
#         file_path.write_text(json.dumps(data, indent=2))

# # Initialize data manager
# data_manager = CustomerDataManager(DATA_DIR)

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     session = request.session
#     if 'conversation_history' not in session:
#         session['conversation_history'] = []
#         session['user_info'] = {}
#         # Start the conversation with a warm welcome
#         initial_message = {
#             "role": "assistant",
#             "content": "Welcome to Tyler-B Car Dealership! I'm Alex, your assistant. Are you looking to buy or lease a car today?"
#         }
#         session['conversation_history'].append(initial_message)
#     return templates.TemplateResponse(
#         "index.html",
#         {"request": request, "initial_message": session['conversation_history'][0]['content']}
#     )

# @app.get('/favicon.ico')
# async def favicon():
#     return FileResponse('static/favicon.ico', media_type='image/vnd.microsoft.icon')

# async def process_message(message: str, user_info: dict, conversation_history: list) -> str:
#     # Log user input to the terminal
#     print(f"User Input: {message}")

#     # Update user information based on the flow of the conversation
#     if len(user_info) == 0:  # First interaction after the welcome
#         user_info['looking_for'] = message.strip()
#     elif 'vehicle_type' not in user_info and any(word in message.lower() for word in ['sedan', 'suv', 'truck', 'van']):
#         user_info['vehicle_type'] = message
#     elif 'budget' not in user_info and any(word in message.lower() for word in ['$', 'dollar', 'budget', 'spend']):
#         user_info['budget'] = message
#     elif 'specific_model' not in user_info and any(word in message.lower() for word in ['x3', 'x5', 'x1', 'model']):
#         user_info['specific_model'] = message

#     # Create a dynamic system prompt based on updated user info
#     messages = [
#         {"role": "system", "content": SYSTEM_CONTEXT},
#         {"role": "system", "content": f"Current user info: {json.dumps(user_info)}"}
#     ]
#     messages.extend(conversation_history[-3:])  # Limit to the last 3 interactions for context

#     input_text = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])

#     try:
#         response = await asyncio.to_thread(model.invoke, input_text)
#         # Log bot response to the terminal
#         print(f"Bot Output: {response}")
#         return response
#     except Exception as e:
#         logger.error(f"Model error: {e}")
#         return "I'm sorry, but I couldn't process that. Could you rephrase?"

# @app.post("/api")
# async def api(request: Request):
#     try:
#         data = await request.json()
#         message = data.get("message")
#         if not message:
#             raise HTTPException(status_code=400, detail="Message is required")

#         session = request.session
#         session.setdefault('conversation_history', [])
#         session.setdefault('user_info', {})

#         session['conversation_history'].append({"role": "user", "content": message})

#         response = await process_message(
#             message,
#             session['user_info'],
#             session['conversation_history']
#         )

#         session['conversation_history'].append({
#             "role": "assistant",
#             "content": response
#         })

#         data_manager.save_interaction(
#             str(id(session)),
#             session['user_info'],
#             session['conversation_history']
#         )

#         return JSONResponse({"content": response})

#     except Exception as e:
#         logger.error(f"Error processing request: {e}", exc_info=True)
#         return JSONResponse(
#             {"error": "An error occurred processing your request"}, 
#             status_code=500
#         )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)






################################################################

# working project 2 

#################################################################


# from fastapi import FastAPI, Request
# from fastapi.responses import HTMLResponse, JSONResponse
# from fastapi.templating import Jinja2Templates
# from fastapi.staticfiles import StaticFiles
# from fastapi.middleware.cors import CORSMiddleware
# from langchain_ollama import OllamaLLM
# import logging
# import json
# import re
# import os
# from difflib import get_close_matches
# import asyncio
# from pathlib import Path


# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger("car_dealership_chatbot")

# app = FastAPI()
# templates = Jinja2Templates(directory="templates")
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # CORS setup
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # Initialize Llama3 model
# llm_model = OllamaLLM(
#     model="llama3",
#     temperature=0.7,
#     max_tokens=500
# )

# # Initialize Jinja2 templates
# templates = Jinja2Templates(directory="templates")


# # System Context
# SYSTEM_CONTEXT = '''You are an AI assistant for a car dealership. Your role is to help customers find their ideal vehicle based on their preferences and requirements.

# Guidelines:
# 1. Respond conversationally and professionally.
# 2. Gather key details efficiently:
#    - Buy or lease preference
#    - Vehicle type (sedan, SUV, truck, etc.)
#    - Budget or payment preference
#    - Specific features or requirements
# 3. Ask questions one by one to avoid overwhelming the user.
# 4. Provide up to three recommendations initially.
# 5. Adjust recommendations dynamically based on user feedback.
# 6. End conversations gracefully when the user indicates they’re finished.

# Maintain the following conversation structure:
# 1. Gather missing information if necessary.
# 2. If the user provides all necessary information, provide recommendations.
# 3. If the user asks for more details, provide detailed information about the selected vehicle.
# 4. If the user asks about deals or financing, provide relevant information.
# 5. If the user doesn’t want any specific features, leave it to none.
# 6. Acknowledge provided preferences.
# 7. Provide concise and helpful responses.
# 8. Follow up with relevant next steps or queries.

# When providing recommendations:
# - Highlight key features, price, and financing/lease options.
# - Present vehicles sorted by relevance to the user's preferences.
# - Offer additional details or alternatives if asked.

# Always remember:
# - Avoid redundant questions.
# - Remember previous interactions and user preferences to provide personalized recommendations.
# - Don’t repeat the same information multiple times.
# - Don’t repeat the same questions multiple times.
# - Keep responses short but informative.
# - Summarize the key points for the user.
# - Then show recommendations based on the user preferences.
# - Be polite and professional.'''


# class ConversationManager:
#     def __init__(self):
#         self.inventory = CarInventory()

#     def determine_budget_category(self, budget_str: str) -> str:
#         try:
#             budget = float(re.sub(r"[^\d.]", "", budget_str))
#             return "economy" if budget <= 35000 else "luxury"
#         except ValueError:
#             raise ValueError("Invalid budget format. Could not parse numeric value.")

#     def find_closest_vehicle(self, message: str, vehicles: list) -> str:
#         vehicle_names = [vehicle["name"] for vehicle in vehicles]
#         closest_matches = get_close_matches(
#             message.lower(), [name.lower() for name in vehicle_names], n=1, cutoff=0.5
#         )
#         if closest_matches:
#             match_index = [name.lower() for name in vehicle_names].index(closest_matches[0])
#             return vehicle_names[match_index]
#         return None

#     def save_user_preferences(self, user_info: dict):
#         """Save user preferences to a JSON file."""
#         try:
#             os.makedirs("data", exist_ok=True)
#             with open("data/user_preferences.json", "w") as file:
#                 json.dump(user_info, file, indent=2)
#         except Exception as e:
#             logger.error(f"Failed to save user preferences: {e}")

#     def load_user_preferences(self) -> dict:
#         """Load user preferences from a JSON file."""
#         try:
#             with open("data/user_preferences.json", "r") as file:
#                 return json.load(file)
#         except FileNotFoundError:
#             logger.info("User preferences file not found. Starting fresh.")
#             return {}
#         except Exception as e:
#             logger.error(f"Failed to load user preferences: {e}")
#             return {}



# class CarInventory:
#     def __init__(self):
#         self.inventory = {
#             "sedan": {
#                 "economy": [
#                     {
#                         "name": "Toyota Camry",
#                         "price": 25000,
#                         "lease": 300,
#                         "features": ["Bluetooth", "Backup Camera", "Lane Departure Warning"],
#                         "deals": ["0% APR for 60 months", "$1500 cash back", "Free maintenance for 2 years"]
#                     },
#                     {
#                         "name": "Honda Accord",
#                         "price": 26000,
#                         "lease": 320,
#                         "features": ["Adaptive Cruise Control", "Apple CarPlay", "Blind Spot Monitor"],
#                         "deals": ["1.9% APR for 72 months", "$2000 cash back", "No payments for 90 days"]
#                     },
#                     {
#                         "name": "Hyundai Sonata",
#                         "price": 24000,
#                         "lease": 290,
#                         "features": ["Panoramic Sunroof", "Wireless Charging", "Remote Start"],
#                         "deals": ["0.9% APR for 48 months", "$1000 loyalty bonus", "Complimentary service package"]
#                     }
#                 ],
#                 "luxury": [
#                     {
#                         "name": "BMW 3 Series",
#                         "price": 42000,
#                         "lease": 550,
#                         "features": ["Leather Seats", "Sunroof", "Premium Sound System"],
#                         "deals": ["2.9% APR for 36 months", "First 3 payments waived", "Complimentary maintenance package"]
#                     },
#                     {
#                         "name": "Mercedes C-Class",
#                         "price": 45000,
#                         "lease": 580,
#                         "features": ["Premium Audio", "Navigation", "Driver Assistance Package"],
#                         "deals": ["3.9% APR financing", "$3000 off MSRP", "4 years free maintenance"]
#                     }
#                 ]
#             },
#             "suv": {
#                 "economy": [
#                     {
#                         "name": "Toyota RAV4",
#                         "price": 28000,
#                         "lease": 350,
#                         "features": ["All-Wheel Drive", "Lane Assist", "Safety Sense 2.0"],
#                         "deals": ["1.9% APR for 60 months", "$2500 cash back", "Free winter tire package"]
#                     },
#                     {
#                         "name": "Honda CR-V",
#                         "price": 29000,
#                         "lease": 370,
#                         "features": ["Heated Seats", "Apple CarPlay", "Honda Sensing"],
#                         "deals": ["0.9% APR for 48 months", "$2000 trade-in bonus", "Extended warranty"]
#                     },
#                     {
#                         "name": "Mazda CX-5",
#                         "price": 27000,
#                         "lease": 340,
#                         "features": ["Blind Spot Monitoring", "Bose Audio", "Power Liftgate"],
#                         "deals": ["0% APR for 36 months", "$1500 bonus cash", "First payment waived"]
#                     }
#                 ],
#                 "luxury": [
#                     {
#                         "name": "Lexus RX",
#                         "price": 50000,
#                         "lease": 600,
#                         "features": ["Premium Sound System", "Navigation", "Leather Interior"],
#                         "deals": ["1.9% APR luxury financing", "Complimentary maintenance", "Lease loyalty bonus"]
#                     },
#                     {
#                         "name": "BMW X5",
#                         "price": 55000,
#                         "lease": 650,
#                         "features": ["Panoramic Roof", "Gesture Control", "Executive Package"],
#                         "deals": ["2.9% APR for 72 months", "Summer sales event pricing", "BMW Ultimate Care"]
#                     }
#                 ]
#             }
#         }

#     def get_vehicles(self, vehicle_type: str, budget_category: str):
#         return self.inventory.get(vehicle_type, {}).get(budget_category, [])

#     def get_vehicle_details(self, vehicle_name: str):
#         for category in self.inventory.values():
#             for budget, vehicles in category.items():
#                 for vehicle in vehicles:
#                     if vehicle_name.lower() in vehicle["name"].lower():
#                         return vehicle
#         return None



# # Process Message Function
# async def process_message(message: str, conversation_data: dict) -> dict:
#     try:
#         state = conversation_data.get("state", "greeting")
#         user_info = conversation_data.get("user_info", {})
#         vehicles = conversation_data.get("vehicles", [])

#         manager = ConversationManager()
#         logger.info(f"Processing message with state: {state}, message: '{message}'")

#         # Default response and next state
#         response = ""
#         next_state = "greeting"

#         # Construct the prompt based on state
#         if state == "greeting":
#             prompt = "You are an AI assistant. Greet the user and ask if they want to buy or lease a car."
#             next_state = "get_intent"

#         elif state == "get_intent":
#             prompt = f"""
#             The user said: '{message}'. Determine if they want to buy or lease a car.
#             Respond accordingly and ask them about the type of car they are interested in.
#             """
#             next_state = "get_vehicle_type"

#         elif state == "get_vehicle_type":
#             vehicle_type = next(
#                 (vtype for vtype in ["sedan", "suv", "truck", "van"] if vtype in message.lower()), None
#             )
#             if vehicle_type:
#                 user_info["vehicle_type"] = vehicle_type
#                 prompt = f"""
#                 The user has selected a {vehicle_type.title()}. Ask them for their budget range.
#                 """
#                 next_state = "get_budget"
#             else:
#                 prompt = "The user has not specified a valid vehicle type. Ask them to clarify their choice."
#                 next_state = state

#         elif state == "get_budget":
#             try:
#                 user_info["budget_category"] = manager.determine_budget_category(message)
#                 vehicles = manager.inventory.get_vehicles(
#                     user_info["vehicle_type"], user_info["budget_category"]
#                 )
#                 conversation_data["vehicles"] = vehicles
#                 if vehicles:
#                     prompt = f"""
#                     Based on their preferences, suggest some vehicles and ask if they would like detailed recommendations.
#                     """
#                     next_state = "show_recommendations"
#                 else:
#                     prompt = "There are no vehicles in their budget range. Ask if they would like to adjust their budget."
#                     next_state = state
#             except ValueError:
#                 prompt = "The user's budget input is invalid. Ask them to specify a budget in numeric terms."
#                 next_state = state

#         elif state == "show_recommendations":
#             prompt = """
#             Provide the user with detailed recommendations for vehicles. Include price, lease, features, and deals.
#             """
#             next_state = "get_confirmation"

#         elif state == "get_confirmation":
#             vehicle_name = manager.find_closest_vehicle(message, vehicles)
#             if vehicle_name:
#                 vehicle_details = manager.inventory.get_vehicle_details(vehicle_name)
#                 user_info["selected_vehicle"] = vehicle_details
#                 prompt = f"""
#                 Provide detailed information about the {vehicle_name}, including:
#                 - Price: ${vehicle_details['price']}
#                 - Lease: ${vehicle_details['lease']}
#                 - Features: {', '.join(vehicle_details['features'])}
#                 - Deals: {', '.join(vehicle_details['deals'])}.
#                 Ask if they would like to proceed with this vehicle.
#                 """
#                 next_state = "final_confirmation"
#             else:
#                 prompt = "The user's message did not match any vehicle. Ask them to clarify."
#                 next_state = "show_recommendations"

#         elif state == "final_confirmation":
#             if "yes" in message.lower():
#                 selected_vehicle = user_info.get("selected_vehicle", {})
#                 if selected_vehicle:
#                     prompt = f"""
#                     The user has chosen to proceed with the {selected_vehicle['name']}. Confirm their choice and ask if they
#                     would like to speak with a sales representative.
#                     """
#                     next_state = "end"
#                 else:
#                     prompt = "No vehicle has been finalized yet. Ask them to explore more options."
#                     next_state = "get_vehicle_type"
#             else:
#                 prompt = "Ask the user if they would like to explore other options or adjust their preferences."
#                 next_state = "get_vehicle_type"

#         else:
#             prompt = "The user's query could not be processed. Ask them to clarify or provide more details."
#             next_state = "greeting"

#         # Generate response using Llama3 model
#         ai_response = await asyncio.to_thread(llm_model.invoke, prompt)

#         logger.info(f"AI Response: {ai_response.strip()}")

#         return {
#             "content": ai_response.strip(),
#             "state": next_state,
#             "user_info": user_info,
#             "vehicles": vehicles,
#         }

#     except Exception as e:
#         logger.error(f"Error processing message: {e}", exc_info=True)
#         return {
#             "content": "Apologies, something went wrong. Could you rephrase or try again?",
#             "state": conversation_data.get("state", "greeting"),
#         }

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})
# @app.post("/api")
# async def chat_endpoint(request: Request):
#     try:
#         data = await request.json()
#         message = data.get("message", "")
#         conversation_data = data.get("conversation_data", {})

#         result = await process_message(message, conversation_data)
#         return JSONResponse(result)

#     except json.JSONDecodeError:
#         logger.error("Invalid JSON in request.", exc_info=True)
#         return JSONResponse(
#             {"error": "The request contains invalid JSON. Please ensure the format is correct."},
#             status_code=400
#         )
#     except Exception as e:
#         logger.error(f"API Error: {e}", exc_info=True)
#         return JSONResponse(
#             {"error": "I encountered an issue processing your request. Please try again later."},
#             status_code=500
#         )

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)




###############################################################



#working project 3

################################################################
# import openai
# from pydantic import BaseModel  # Import BaseModel from pydantic
# from fastapi import FastAPI, HTTPException, Request
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from fastapi.middleware.cors import CORSMiddleware
# from langchain_ollama import OllamaLLM
# import asyncio
# import os

# # Initialize FastAPI app
# app = FastAPI()


# # Middleware configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust as needed
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Templates directory for HTML files
# templates = Jinja2Templates(directory="templates")

# # Initialize LLaMA model
# llm_model = OllamaLLM(model="llama3", temperature=0.7, max_tokens=512)
# #openai.api_key = "your-openai-api-key-here"


# # System prompt for the chatbot’s behavior
# SYSTEM_PROMPT = (
#     "You are a friendly, professional car dealership assistant. Help the user find a car based on their preferences. "
#     "You will ask the user for their desired car action (buy/lease), car type (e.g., sedan, SUV), car make (e.g., Toyota, Honda), "
#     "car condition (new or used), budget, and any special requirements. Be sure to respond in a natural and understanding way. "
#     "Ensure that you guide the user step-by-step to get all the information needed for car recommendations."
# )

# # Define Pydantic model to collect user input for car buying/leasing
# class CarDealershipInput(BaseModel):
#     action: str  # 'buy' or 'lease'
#     car_type: str  # 'sedan', 'SUV', etc.
#     car_make: str  # 'Toyota', 'Honda', etc.
#     car_condition: str  # 'new' or 'used'
#     budget: float  # User's budget
#     special_requirements: str = None  # Any special requirements (optional)

# # Initialize conversation state
# conversation_state = {
#     'state': "greeting",  # Starting state
#     'user_info': {}
# }

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     """
#     Serve the HTML page for the chat interface.
#     """
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.post("/api")
# async def process_message(data: dict):
#     """
#     Process the user input message and return a structured response based on the conversation flow.
#     """
#     user_message = data.get('message')
    
#     # Log user input for debugging
#     print(f"User Input: {user_message}")

#     # Process the input based on current state
#     if conversation_state['state'] == "greeting":
#         # Ask buy/lease
#         response = await get_llama_response("Would you like to buy or lease a car?")
#         conversation_state['state'] = "action"
#         return {"content": response, "state": conversation_state['state'], "user_info": conversation_state['user_info']}

#     elif conversation_state['state'] == "action":
#         # Handle invalid or valid buy/lease input
#         action = user_message.strip().lower()
#         if action in ["buy", "lease"]:
#             conversation_state['user_info']['action'] = action
#             conversation_state['state'] = "car_type"
#             response = await get_llama_response(f"Great! Are you looking for a {action} car? Please specify the type (e.g., sedan, SUV).")
#         else:
#             response = await get_llama_response("I didn't quite understand. Do you want to buy or lease a car?")
#         return {"content": response, "state": conversation_state['state'], "user_info": conversation_state['user_info']}

#     elif conversation_state['state'] == "car_type":
#         # Get car type
#         car_type = user_message.strip().lower()
#         conversation_state['user_info']['car_type'] = car_type
#         conversation_state['state'] = "car_make"
#         response = await get_llama_response(f"Got it! You want a {car_type}. What make of car are you interested in (e.g., Toyota, Honda)?")
#         return {"content": response, "state": conversation_state['state'], "user_info": conversation_state['user_info']}

#     elif conversation_state['state'] == "car_make":
#         # Get car make
#         car_make = user_message.strip().lower()
#         conversation_state['user_info']['car_make'] = car_make
#         conversation_state['state'] = "car_condition"
#         response = await get_llama_response(f"Perfect! What about the condition? Do you prefer a new or used {car_make}?")
#         return {"content": response, "state": conversation_state['state'], "user_info": conversation_state['user_info']}

#     elif conversation_state['state'] == "car_condition":
#         # Handle car condition
#         car_condition = user_message.strip().lower()
#         if car_condition in ["new", "used"]:
#             conversation_state['user_info']['car_condition'] = car_condition
#             conversation_state['state'] = "budget"
#             response = await get_llama_response(f"Great choice! What's your budget for a {car_condition} {conversation_state['user_info']['car_type']}?")
#         else:
#             response = await get_llama_response("Please specify if you want a new or used car.")
#         return {"content": response, "state": conversation_state['state'], "user_info": conversation_state['user_info']}

#     elif conversation_state['state'] == "budget":
#         # Handle invalid or valid budget input
#         try:
#             budget = float(user_message.strip())
#             conversation_state['user_info']['budget'] = budget
#             conversation_state['state'] = "special_requirements"
#             response = await get_llama_response(f"Thanks! Do you have any special requirements for the car?")
#         except ValueError:
#             response = await get_llama_response("Please provide a valid budget amount.")
#         return {"content": response, "state": conversation_state['state'], "user_info": conversation_state['user_info']}

#     elif conversation_state['state'] == "special_requirements":
#         # Get special requirements
#         special_requirements = user_message.strip()
#         conversation_state['user_info']['special_requirements'] = special_requirements
#         conversation_state['state'] = "summary"
#         summary = summarize_requirements(conversation_state['user_info'])
#         cars_found = search_inventory(conversation_state['user_info'])
#         response = await get_llama_response(f"Here is a summary of your preferences:\n{summary}\nLet me find some cars for you.")
#         return {
#             "content": response,
#             "state": conversation_state['state'],
#             "user_info": conversation_state['user_info'],
#             "cars": cars_found
#         }

#     elif conversation_state['state'] == "summary":
#         # Show inventory matches
#         return {"content": "I found these cars for you!", "state": conversation_state['state'], "user_info": conversation_state['user_info'], "cars": []}

# async def get_llama_response(user_input: str) -> str:
#     """
#     Use LLaMA3 to generate a refined response based on the user input.
#     """
#     prompt = SYSTEM_PROMPT + "\nUser input: " + user_input
#     try:
#         response = await llm_model.acomplete(prompt)
#         return response.text.strip()
#     except Exception as e:
#         print(f"Error with LLaMA response: {e}")
#         return "I'm having trouble understanding. Could you please clarify?"

# def summarize_requirements(user_info: dict) -> str:
#     """
#     Summarize the user's requirements based on their input.
#     """
#     return (
#         f"User wants to {user_info['action']} a {user_info['car_condition']} {user_info['car_type']} from {user_info['car_make']} "
#         f"with a budget of {user_info['budget']} and the following special requirements: {user_info.get('special_requirements', 'None')}."
#     )

# def search_inventory(user_info: dict):
#     """
#     Mock function to search the car inventory based on user input.
#     """
#     # Example inventory (replace with actual database/API integration)
#     inventory = [
#         {"make": "Toyota", "model": "Camry", "type": "sedan", "condition": "new", "price": 25000},
#         {"make": "Honda", "model": "Civic", "type": "sedan", "condition": "used", "price": 15000},
#         {"make": "Ford", "model": "Explorer", "type": "SUV", "condition": "new", "price": 35000},
#         {"make": "BMW", "model": "X5", "type": "SUV", "condition": "used", "price": 30000},
#     ]
    
#     # Filter the inventory based on user input
#     filtered_cars = [
#         car for car in inventory
#         if car["type"].lower() == user_info['car_type'].lower() and
#            car["make"].lower() == user_info['car_make'].lower() and
#            car["condition"].lower() == user_info['car_condition'].lower() and
#            car["price"] <= user_info['budget']
#     ]
    
#     return filtered_cars

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)

################################################################
# working project 4
################################################################

# from fastapi import FastAPI, HTTPException, Request
# from fastapi.responses import HTMLResponse, FileResponse
# from fastapi.staticfiles import StaticFiles
# from fastapi.templating import Jinja2Templates
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel
# from langchain_ollama import OllamaLLM
# import asyncio
# import re

# # Initialize FastAPI app
# app = FastAPI()

# # Middleware configuration
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],  # Adjust as needed
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# # Mount static files (CSS/JS)
# app.mount("/static", StaticFiles(directory="static"), name="static")

# # Templates directory for HTML files
# templates = Jinja2Templates(directory="templates")

# # Initialize LLaMA model
# llm_model = OllamaLLM(model="llama3", temperature=0.7, max_tokens=512)

# # Pydantic model for input validation
# class LLMInput(BaseModel):
#     input_text: str

# # System prompt for professional, clear responses without special characters
# SYSTEM_PROMPT = (
#     "You are a professional and friendly assistant. Always provide clear, concise, and precise answers. "
#     "Organize your responses in plain text without using special characters like asterisks (**), dashes (-), or similar. "
#     "Focus on delivering straightforward, easy-to-read information in well-structured sentences and paragraphs."
# )

# # Store conversation history globally (or use a more persistent storage solution as needed)
# conversation_history = []

# # Track user information and conversation state
# user_info = {
#     "buy_lease": None,
#     "car_type": None,
#     "car_maker": None,
#     "car_condition": None,
#     "budget": None,
#     "special_requirements": None
# }

# # List of sample cars in inventory (simulated for the example)
# inventory = [
#     {"make": "Toyota", "type": "Sedan", "condition": "new", "price": 30000, "features": "Bluetooth, Backup Camera"},
#     {"make": "Ford", "type": "SUV", "condition": "used", "price": 15000, "features": "Sunroof, Leather Seats"},
#     {"make": "Honda", "type": "Coupe", "condition": "new", "price": 25000, "features": "Apple CarPlay, Heated Seats"},
#     {"make": "BMW", "type": "Sedan", "condition": "new", "price": 45000, "features": "Navigation, Heated Seats"},
#     {"make": "BMW", "type": "SUV", "condition": "new", "price": 50000, "features": "Leather Seats, Backup Camera, Sunroof"}
# ]

# def format_response(response: str) -> str:
#     """
#     Format the response to ensure consistency and better readability.
#     """
#     response = response.strip()
#     if not response.endswith("."):
#         response += "."
#     return response.capitalize()

# def search_inventory(user_info):
#     """
#     Simulate searching for cars in the inventory based on the user's criteria.
#     """
#     filtered_cars = []
#     for car in inventory:
#         if (user_info["car_type"] == car["type"] and
#             user_info["car_maker"].lower() in car["make"].lower() and
#             user_info["car_condition"] == car["condition"] and
#             car["price"] <= user_info["budget"]):
#             filtered_cars.append(car)
#     return filtered_cars


# def parse_budget(input_text):
#     """
#     Parse the budget input to handle various formats (e.g., "$30000", "around 30k").
#     """
#     match = re.search(r"(\d+[\.,]?\d*)\s*(dollars?|k|usd|million)?", input_text.lower())
#     if match:
#         budget = match.group(1)
#         if "k" in match.group(2):
#             budget = float(budget) * 1000
#         elif "million" in match.group(2):
#             budget = float(budget) * 1000000
#         return int((budget))  # Return as integer
#     else:
#         return None  # Return None if no valid budget is found

# @app.get("/", response_class=HTMLResponse)
# async def index(request: Request):
#     """
#     Serve the HTML page for the chat interface.
#     """
#     return templates.TemplateResponse("index.html", {"request": request})

# @app.get("/favicon.ico")
# async def favicon():
#     """
#     Serve the favicon.ico file.
#     """
#     return FileResponse("static/favicon.ico")

# @app.post("/api/")
# async def process_with_llm(data: LLMInput):
#     """
#     Process input text and return a structured response from the LLaMA model.
#     """
#     global conversation_history, user_info
#     try:
#         # Log user input in the terminal
#         print(f"User Input: {data.input_text}")
        
#         # Add user input to the conversation history
#         conversation_history.append(f"User: {data.input_text}")

#         # Handle conversation stages, ensuring that LLaMA introduces itself at the start
#         if not conversation_history:
#             introduction = "Hello! I am your Car Dealership Assistant. I can help you with buying or leasing a car."
#             conversation_history.append(f"Assistant: {introduction}")
#             response = introduction
#         else:
#             current_stage = None
#             if user_info["buy_lease"] is None:
#                 current_stage = "buy_lease"
#             elif user_info["car_type"] is None:
#                 current_stage = "car_type"
#             elif user_info["car_maker"] is None:
#                 current_stage = "car_maker"
#             elif user_info["car_condition"] is None:
#                 current_stage = "car_condition"
#             elif user_info["budget"] is None:
#                 current_stage = "budget"
#             elif user_info["special_requirements"] is None:
#                 current_stage = "special_requirements"
#             else:
#                 current_stage = "summary"

#             # Ask questions based on the current stage of the conversation
#             if current_stage == "buy_lease":
#                 user_info["buy_lease"] = data.input_text
#                 response = "Are you interested in a sedan, SUV, or another type of car?"
#             elif current_stage == "car_type":
#                 user_info["car_type"] = data.input_text
#                 response = "What maker (brand) of car are you interested in?"
#             elif current_stage == "car_maker":
#                 user_info["car_maker"] = data.input_text
#                 response = "Are you looking for a new or used car?"
#             elif current_stage == "car_condition":
#                 user_info["car_condition"] = data.input_text
#                 response = "What is your budget range for the car?"
#             elif current_stage == "budget":
#                 parsed_budget = parse_budget(data.input_text)
#                 if parsed_budget:
#                     user_info["budget"] = parsed_budget
#                     response = "Do you have any special requirements for the car?"
#                 else:
#                     response = "Please enter a valid number for your budget, or provide a range (e.g., $30,000)."
#             elif current_stage == "special_requirements":
#                 user_info["special_requirements"] = data.input_text
#                 response = f"Here’s a summary of your preferences: \n" \
#                            f"Buy/Lease: {user_info['buy_lease']}, Type: {user_info['car_type']}, " \
#                            f"Maker: {user_info['car_maker']}, Condition: {user_info['car_condition']}, " \
#                            f"Budget: {user_info['budget']}, Special Requirements: {user_info['special_requirements']}.\n" \
#                            f"Let me find some options for you."

#                 # Simulate searching the inventory based on the user's preferences
#                 cars_found = await asyncio.to_thread(search_inventory, user_info)  # Run this in a separate thread
#                 if cars_found:
#                     response += "\nHere are some cars that match your preferences:\n"
#                     for car in cars_found:
#                         response += f"Make: {car['make']}, Type: {car['type']}, Condition: {car['condition']}, " \
#                                     f"Price: {car['price']}, Features: {car['features']}\n"
#                 else:
#                     response += "\nSorry, no cars match your criteria."

#                 # End the conversation
#                 conversation_history.clear()
#                 user_info.clear()

#         # Add assistant's response to the conversation history
#         conversation_history.append(f"Assistant: {response}")
        
#         # Log bot output in the terminal
#         print(f"Bot Response: {response}")
        
#         if response:
#             return {"input": data.input_text, "response": response}
#         else:
#             return {"input": data.input_text, "response": "I couldn't process that. Could you please rephrase?"}
#     except Exception as e:
#         # Log the error in the terminal
#         print(f"Error processing text: {e}")
#         raise HTTPException(status_code=500, detail=f"Error processing text: {e}")

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run(app, host="0.0.0.0", port=8000)



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