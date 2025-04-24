import openai
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from datetime import datetime
import os

@dataclass
class CustomerInfo:
    # Personal Information
    first_name: Optional[str] = None
    last_name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    zip: Optional[str] = None

    # Car Type Selection
    car_condition: Optional[str] = None  # 'new' or 'pre-owned'
    transaction_type: Optional[str] = None  # 'buy' or 'lease'

    # Purchase Information (only for buying)
    payment_method: Optional[str] = None
    credit_rating: Optional[str] = None
    max_monthly_payment: Optional[str] = None
    max_down_payment: Optional[str] = None
    max_finance_term: Optional[str] = None

    # Lease Information (only for leasing)
    lease_term: Optional[str] = None
    annual_mileage: Optional[str] = None
    lease_down_payment: Optional[str] = None

    # Car Information
    car_type: Optional[str] = None
    make: Optional[str] = None
    year: Optional[str] = None
    model: Optional[str] = None
    max_mileage: Optional[str] = None  # Only for pre-owned cars
    desired_features: Optional[str] = None

    # Trade-in Information
    has_trade_in: Optional[bool] = None

    # Budget & Additional Information
    min_budget: Optional[str] = None
    max_budget: Optional[str] = None


class CarSalesGPTBot:
    def __init__(self, api_key):
        openai.api_key = api_key
        self.customer = CustomerInfo()
        self.conversation_history = []
        self.current_collection_phase = "personal_info"
        self.all_information_collected = False
        self.conversation_file = None
        self.required_fields = {
            'personal_info': ['first_name', 'last_name', 'email', 'phone', 'zip'],
            'budget': ['min_budget', 'max_budget', 'credit_rating'],
            'car_selection': ['car_condition', 'transaction_type'],
            'car_details': ['car_type', 'make', 'model', 'year', 'desired_features'],
            'trade_in': ['has_trade_in'],
            'transaction_details': []  # Will be populated based on transaction_type
        }

    def update_required_fields(self):
        """Update only transaction_details based on current state, ensuring correct logic for new vs pre-owned cars."""
        transaction_type = self.customer.transaction_type
        car_condition = self.customer.car_condition

        # If the car is pre-owned, force transaction type to 'buy'
        if car_condition == 'pre-owned':
            self.customer.transaction_type = 'buy'
            transaction_type = 'buy'

        # Ensure mutual exclusivity between buy and lease options
        if transaction_type == 'buy':
            self.required_fields['transaction_details'] = [
                'payment_method',
                'max_monthly_payment',
                'max_down_payment',
                'max_finance_term'
            ]
            # Reset lease-related fields
            self.customer.lease_term = "N/A"
            self.customer.annual_mileage = "N/A"
            self.customer.lease_down_payment = "N/A"
        elif transaction_type == 'lease':
            self.required_fields['transaction_details'] = [
                'lease_term',
                'annual_mileage',
                'lease_down_payment'
            ]
            # Reset buy-related fields
            self.customer.payment_method = "N/A"
            self.customer.max_monthly_payment = "N/A"
            self.customer.max_down_payment = "N/A"
            self.customer.max_finance_term = "N/A"
        else:
            self.required_fields['transaction_details'] = []

    def get_missing_fields(self):
        """Return the list of missing fields that need to be collected, treating 'N/A' as a valid response."""
        current_fields = self.required_fields.get(self.current_collection_phase, [])
        return [
            field for field in current_fields
            if getattr(self.customer, field) in (None, "")  # Allow 'N/A' as valid
        ]

    def check_phase_completion(self):
        """Check if the current phase is complete and move to the next phase if needed, ensuring all phases are covered."""
        missing_fields = self.get_missing_fields()
        
        if not missing_fields:  # If no missing fields, move forward
            next_phase = self._get_next_phase()
            if next_phase:
                self.current_collection_phase = next_phase
                print(f"✅ Moving to next phase: {self.current_collection_phase}")
            else:
                self.all_information_collected = True
                print("✅ All phases completed successfully.")
            return False
        
        return False


    def _get_next_phase(self):
        """Get the next phase in the collection process."""
        phases = list(self.required_fields.keys())
        current_index = phases.index(self.current_collection_phase)
        if current_index < len(phases) - 1:
            return phases[current_index + 1]
        return None

    def process_message(self, user_message):
        """Process user message and update fields accordingly."""
        try:
            self.initialize_conversation_file()
            self.conversation_history.append({"role": "user", "content": user_message})
            extracted_info = self.extract_information(user_message)
            self.update_customer_info(extracted_info)
            self.update_required_fields()
            response = self.get_bot_response()
            self.conversation_history.append({"role": "assistant", "content": response})
            self.save_conversation()
            return response
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            self.save_conversation()
            return error_msg

  

    def update_customer_info(self, extracted_info: Dict):
        """
        Update customer information with validated data, ensuring proper type conversion
        and handling of special values.
        """
        for field, value in extracted_info.items():
            if hasattr(self.customer, field):
                current_value = getattr(self.customer, field)
                
                # Special handling for trade-in boolean field
                if field == 'has_trade_in':
                    if isinstance(value, bool):
                        setattr(self.customer, field, value)
                    elif isinstance(value, str):
                        # Convert string responses to boolean
                        if value.lower() in ['yes', 'true', 'y', '1']:
                            setattr(self.customer, field, True)
                        elif value.lower() in ['no', 'false', 'n', '0']:
                            setattr(self.customer, field, False)
                    continue
                
                # Handle "N/A" values
                if isinstance(value, str) and value.lower() in ["no", "none", "i don't have any"]:
                    value = "N/A"
                
                # Handle numeric fields (budgets, payments, etc.)
                if field in ['min_budget', 'max_budget', 'max_monthly_payment', 'max_down_payment']:
                    try:
                        if isinstance(value, str):
                            # Remove currency symbols and convert k to thousands
                            value = value.replace('$', '').replace(',', '')
                            if 'k' in value.lower():
                                value = float(value.lower().replace('k', '')) * 1000
                            else:
                                value = float(value)
                    except (ValueError, TypeError):
                        continue
                
                # Only update if:
                # 1. The field is currently empty (None)
                # 2. The new value is different from current
                # 3. The new value is not "N/A"
                if value and (current_value is None or 
                             str(value).lower() != str(current_value).lower()) and value != "N/A":
                    setattr(self.customer, field, value)
                    print(f"Updated {field}: {value}")
        
        # After updating, check if we can move to next phase
        self.check_phase_completion()


    def get_bot_response(self):
        """Generate the bot's response with improved flow maintenance."""
        missing_fields = self.get_missing_fields()

        if not missing_fields:
            if not self.check_phase_completion():
                return "Thank you for providing all the details! If you have any further questions, feel free to ask."
            return self.handle_exit()

        next_field = missing_fields[0]

        collected_info = {
            'name': f"{self.customer.first_name} {self.customer.last_name}".strip(),
            'email': self.customer.email,
            'phone': self.customer.phone,
            'zip': self.customer.zip,
            'budget_details': {
                'min_budget': self.customer.min_budget,
                'max_budget': self.customer.max_budget,
                'credit_rating': self.customer.credit_rating
            },
            'car_selection': {
                'car_condition': self.customer.car_condition,
                'transaction_type': self.customer.transaction_type
            },
            'car_details': {
                'make': self.customer.make,
                'model': self.customer.model,
                'type': self.customer.car_type,
                'year': self.customer.year,
                'desired_features': self.customer.desired_features,
                'max_mileage': self.customer.max_mileage,  
            },
            'trade_in': {
                'has_trade_in': self.customer.has_trade_in
            },
            'transaction_details': {
                'payment_method': self.customer.payment_method,
                'max_monthly_payment': self.customer.max_monthly_payment,
                'max_down_payment': self.customer.max_down_payment,
                'max_finance_term': self.customer.max_finance_term,
                'lease_term': self.customer.lease_term,
                'annual_mileage': self.customer.annual_mileage,
                'lease_down_payment': self.customer.lease_down_payment
            },
            'current_phase': self.current_collection_phase
        }

        system_prompt = f"""
        You are a professional car sales assistant.
        Current phase: {self.current_collection_phase}
        Next required field: {next_field}

        Already collected information:
        {json.dumps(collected_info, indent=2)}

        Rules:
        1. Ask specifically for the next missing field: {next_field}
        2. If asking for zip code, specify it should be 5 digits
        3. If asking for budget, ask for specific numbers
        4. If asking for credit rating, list all the valid options (Eg. `Excellent`, `Very Good`, `Good`, `Fair`, `Poor`)
        5. If asking for car condition, allow only 'new' or 'pre-owned' options
        6. If asking for transaction type, allow only 'buy' or 'lease'
        7. If car is pre-owned, transaction type must be 'buy' (not 'lease')
        8. If asking about trade-in, confirm whether the user has a trade-in vehicle
        9. Maintain a natural conversation flow
        10. Reference previously collected information when relevant
        11. Be polite and professional
        12. Keep responses focused and concise
        13. When a phase is complete, automatically move to the next set of questions.
        14. Don't end conversation untill all phase are gone through,
            - `personal_info`
            - `budget`
            - `car_selection` 
            - `car_details`
            - `trade_in`
            - `transaction_details`
        """

        response = openai.ChatCompletion.create(
            model="gpt-3.5-turbo",
            temperature=1.0,
            top_p=0.9,
            max_tokens=3800,
            messages=[{"role": "system", "content": system_prompt},
                    *self.conversation_history]
        )
        return response.choices[0].message.content

    def save_conversation(self):
        """Save the conversation history to a single file."""
        if not self.conversation_file:
            self.initialize_conversation_file()

        try:
            conversation_data = {
                'customer_info': asdict(self.customer),
                'conversation_history': self.conversation_history,
                'timestamp': datetime.now().isoformat(),
                'collection_phase': self.current_collection_phase,
                'all_information_collected': self.all_information_collected
            }

            with open(self.conversation_file, 'w', encoding='utf-8') as f:
                json.dump(conversation_data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save conversation history: {e}")

    def initialize_conversation_file(self):
        """Initialize the conversation file at the start of the session."""
        if not self.conversation_file:
            self.conversation_file = f"conversation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            self.save_conversation()

    def extract_information(self, text):
        """Extract relevant information from customer message with improved validation."""
        try:
            name_context = ""
            if self.customer.first_name or self.customer.last_name:
                name_context = f"""
                Current name information:
                - First name: {self.customer.first_name}
                - Last name: {self.customer.last_name}
                """
            extraction_prompt = f"""
            Extract the relevant information from the customer message and return it as a valid JSON object.
            Current phase: {self.current_collection_phase}
            Current customer info: {json.dumps(asdict(self.customer), indent=2)}
            {name_context}

            Rules for extraction:
            1. For name handling:
            - If only one name is provided and we already have a first_name, treat it as last_name
            - If only one name is provided and we have no names yet, treat it as first_name
            - If full name is provided, split into first_name and last_name correctly

            2. IMPORTANT: Distinguish between different payment fields:
           - 'lease_down_payment': Only for lease transactions, represents initial payment for lease
           - 'max_budget': Overall maximum budget for purchase
           - 'max_down_payment': Down payment for purchase transactions
            
            3. When in transaction_details phase and transaction_type is 'lease':
           - Any mentioned payment amount should be interpreted as 'lease_down_payment' or 'max_monthly_payment' based on which field is discussing.
           - Do not update 'max_budget' or other payment fields
            
            4. Current transaction type is: {self.customer.transaction_type}
            
            5. Current collection phase is: {self.current_collection_phase}
            
            6. Ensure `car_condition` is either "new" or "pre-owned"
            
            7. Ensure `transaction_type` is either "buy" or "lease"
            
            8. Extract car details accurately, including `car_make`, `car_model`, `car_type` and `year`.
            
            9. Ensure `year` is a string between 1900 and 2025
            
            10. Ensure `zip` follows the 5-digit format (e.g., 12345)
            
            11. For credit_rating, normalize and map to one of these exact values:
           - "Excellent" for: excellent, perfect, outstanding, exceptional
           - "Very Good" for: very good, very well, great
           - "Good" for: good, okay, fine
           - "Fair" for: fair, average, moderate
           - "Poor" for: poor, bad, low

            If the user responds in a different format, extract and map it to the closest value based on the context. For example:
                - If the user says "it is good", extract and map it as **"Good"**.
                - If the user says "It's excellent", map it to **"Excellent"**.
                - Ensure that the output is a valid value from the list above and use the most appropriate match.
                
            12. Ensure budgets are numeric values
            
            13. For car_type use: "suv", "sedan", "truck", "van", or "coupe"
            
            14. For trade-in information, map common responses to True or False:
            - Normalize responses as follows:
            - **Positive responses** (e.g., "yes", "i have a trade-in", "trade-in available") → **True**
            - **Negative responses** (e.g., "no", "no trade-in", "i don't have a trade-in", "not interested in trade-in") → **False**
            - Handle all variations like "not sure", "maybe", or other ambiguous responses by prompting the user for a clearer answer.
            
            15. If the user does not provide a preference for any field (like last name, car model, etc.), set only that field to "N/A"(Don't change any other fields).
            
            16. Do not extract partial or incomplete information.
            
            17. Do not end conversation without extracting all the necessary information.
            
            18. Always ask the next question don't wait for user response.
            
            19. Understand the user intention to fill field based on the conversation history.

            Message: {text}

            Return a JSON object with only the newly extracted or updated information.
            """
            
            #Perform extraction through GPT-3
            response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0.7,
                top_p=0.9,
                max_tokens=4000,
                messages=[{"role": "system", "content": "You are a precise data extraction assistant. Understand the user input and Extract only the specified information and return it as JSON."},
                        {"role": "user", "content": extraction_prompt}]
            )

            response_content = response.choices[0].message.content.strip()
            print(f"Raw response content: {response_content}")  
            response_content = response_content.replace("```json", "").replace("```", "").strip()
            extracted_info = json.loads(response_content)

            # Handle trade-in response (ensure normalization)
            if "has_trade_in" in extracted_info:
                trade_in_value = extracted_info["has_trade_in"]
                if isinstance(trade_in_value, str):
                    # Normalize responses to True/False
                    if any(word in trade_in_value.lower() for word in ["yes", "have", "available", "trade-in"]):
                        extracted_info["has_trade_in"] = True
                    elif any(word in trade_in_value.lower() for word in ["no", "don't", "none", "not"]):
                        extracted_info["has_trade_in"] = False
                    else:
                        extracted_info.pop("has_trade_in")
                elif not isinstance(trade_in_value, bool):
                    extracted_info.pop("has_trade_in")

            
            # Validate zip code if present
            if "zip" in extracted_info:
                zip_code = str(extracted_info["zip"])
                if not (len(zip_code) == 5 and zip_code.isdigit()):
                    extracted_info.pop("zip")

            # Validate year if present
            if "year" in extracted_info:
                year = str(extracted_info["year"])
                if not (year.isdigit() and 1900 <= int(year) <= 2025):
                    extracted_info.pop("year")

            # Ensure budget values are numeric
            for budget_field in ["min_budget", "max_budget"]:
                if budget_field in extracted_info:
                    try:
                        extracted_info[budget_field] = float(str(extracted_info[budget_field]).replace("k", "000"))
                    except (ValueError, TypeError):
                        extracted_info.pop(budget_field)

            return extracted_info

        except json.JSONDecodeError as e:
            print(f"JSON Decode Error: {e}")
            return {}
        except Exception as e:
            print(f"Warning: Information extraction failed: {e}")
            return {}


    def generate_summary(self):
        """Generate a comprehensive summary of the conversation and customer requirements for dealership.""" 
        try:
            summary_prompt = f"""
            Generate a comprehensive summary of the customer interaction based on the following information:
            
            Customer Information: {json.dumps(asdict(self.customer), indent=2)}

            Please include:
            1. Customer's personal information
            2. Vehicle preferences and requirements
            3. Financial information and budget
            4. Any other additional information gathered from the conversation

            Write down the key notes first.
            Format the summary in a clear, professional manner including all the details gathered from customer personal ,vehicle preferences, requirements, financial information, budget and any other additional information.
            Make it easy to understand and informative .
            """

            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                temperature=1.0,
                top_p=0.9,
                max_tokens=3800,
                messages=[{"role": "system", "content": "You are a professional car sales assistant. Create a detailed summary & general idea of the customer interaction."},
                          {"role": "user", "content": summary_prompt}]
            )

            summary = response.choices[0].message.content
            return summary
        except Exception as e:
            return f"Error generating summary: {str(e)}"

    def generate_summary_to_file(self):
        """Generate a summary and save it to a separate text file.""" 
        try:
            summary = self.generate_summary()
            summary_file = f"conversation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            with open(summary_file, 'w', encoding='utf-8') as f:
                f.write(summary)
            return summary_file
        except Exception as e:
            return f"Error generating summary text file: {str(e)}"

    def handle_exit(self):
        """Handle conversation exit with summary generation.""" 
        if not self.all_information_collected:
            incomplete_fields = self.get_missing_fields()
            summary = f"""
            Conversation ended before all information was collected.
            Missing information: {', '.join(incomplete_fields)}
            """
        else:
            summary = self.generate_summary()

        summary_file = self.generate_summary_to_file()
        return f"Thank you for your time. Here's a summary of our conversation:\n\n{summary}\n\nSummary file saved to {summary_file}\n\nGoodbye!"

def main(history):
    # Initialize the bot with your OpenAI API key
    api_key = "API KEY"  # Replace with your actual OpenAI API key
    bot = CarSalesGPTBot(api_key)

    print("Car Sales Assistant: Hello! I'm here to help you find your perfect vehicle. First, may I have your name?")

    while True:
        try:
            user_input = input("You: ").strip()
            if not user_input:
                print("Car Sales Assistant: I didn't catch that. Could you please repeat?")
                continue

            response = bot.process_message(user_input)
            print(f"Car Sales Assistant: {response}")

            if user_input.lower() in ['/exit']:
                break

        except KeyboardInterrupt:
            print("\nGenerating final summary before exit...")
            print(bot.handle_exit())
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again.")

if __name__ == "__main__": 
    main()


