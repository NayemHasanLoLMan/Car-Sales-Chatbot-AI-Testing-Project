import openai
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict
from datetime import datetime

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
        
        # Define required fields with validation rules
        self.required_fields = {
            'personal_info': ['first_name', 'last_name', 'email', 'phone', 'zip'],
            'budget': ['min_budget', 'max_budget', 'credit_rating'],
            'car_selection': ['car_condition', 'transaction_type'],
            'car_details': ['car_type', 'make', 'model', 'year', 'desired_features'],
            'trade_in': ['has_trade_in'],
            'transaction_details': []  # Will be populated based on transaction_type
        }
        
        # Define validation rules
        self.validation_rules = {
            'zip': lambda x: str(x).isdigit() and len(str(x)) == 5,
            'email': lambda x: '@' in str(x) and '.' in str(x),
            'phone': lambda x: str(x).replace('-', '').replace('(', '').replace(')', '').isdigit(),
            'credit_rating': lambda x: x in ['Excellent', 'Very Good', 'Good', 'Fair', 'Poor'],
            'car_condition': lambda x: x in ['new', 'pre-owned'],
            'transaction_type': lambda x: x in ['buy', 'lease'],
            'car_type': lambda x: x in ['suv', 'sedan', 'truck', 'van', 'coupe'],
            'year': lambda x: str(x).isdigit() and 1900 <= int(x) <= 2025,
            'has_trade_in': lambda x: isinstance(x, bool)
        }
        
        # Define question templates
        self.questions = {
            'first_name': "What is your first name?",
            'last_name': "What is your last name?",
            'email': "What is your email address?",
            'phone': "What is your phone number?",
            'zip': "What is your 5-digit zip code?",
            'min_budget': "What is your minimum budget for the vehicle?",
            'max_budget': "What is your maximum budget for the vehicle?",
            'credit_rating': "How would you rate your credit? (Excellent, Very Good, Good, Fair, or Poor)",
            'car_condition': "Are you interested in a new or pre-owned vehicle?",
            'transaction_type': "Would you like to buy or lease the vehicle?",
            'car_type': "What type of vehicle are you looking for? (SUV, Sedan, Truck, Van, or Coupe)",
            'make': "What make (brand) of vehicle are you interested in?",
            'model': "What model are you interested in?",
            'year': "What year are you looking for?",
            'desired_features': "What features are important to you in the vehicle?",
            'has_trade_in': "Do you have a vehicle to trade in?",
            'payment_method': "How would you like to pay? (Cash, Finance, or Lease)",
            'max_monthly_payment': "What is your maximum monthly payment?",
            'max_down_payment': "What is your maximum down payment?",
            'max_finance_term': "What is your preferred finance term in months?",
            'lease_term': "What lease term would you prefer? (24, 36, or 48 months)",
            'annual_mileage': "How many miles do you expect to drive per year?",
            'lease_down_payment': "What down payment can you make for the lease?",
            'max_mileage': "What is the maximum mileage you'd accept for the pre-owned vehicle?"
        }

    def update_required_fields(self):
        """Update transaction_details based on current state"""
        if self.customer.car_condition == 'pre-owned':
            self.customer.transaction_type = 'buy'
            
        if self.customer.transaction_type == 'buy':
            self.required_fields['transaction_details'] = [
                'payment_method',
                'max_monthly_payment',
                'max_down_payment',
                'max_finance_term'
            ]
            self._reset_lease_fields()
        elif self.customer.transaction_type == 'lease':
            if self.customer.car_condition != 'pre-owned':
                self.required_fields['transaction_details'] = [
                    'lease_term',
                    'annual_mileage',
                    'lease_down_payment'
                ]
                self._reset_buy_fields()

    def _reset_lease_fields(self):
        """Reset lease-related fields"""
        for field in ['lease_term', 'annual_mileage', 'lease_down_payment']:
            setattr(self.customer, field, "N/A")

    def _reset_buy_fields(self):
        """Reset buy-related fields"""
        for field in ['payment_method', 'max_monthly_payment', 'max_down_payment', 'max_finance_term']:
            setattr(self.customer, field, "N/A")

    def get_missing_fields(self):
        """Return list of missing fields for current phase"""
        current_fields = self.required_fields.get(self.current_collection_phase, [])
        return [
            field for field in current_fields
            if getattr(self.customer, field) in (None, "")
        ]

    def validate_field(self, field: str, value: any) -> bool:
        """Validate a field value"""
        if field in self.validation_rules:
            try:
                return self.validation_rules[field](value)
            except Exception:
                return False
        return True

    def get_next_question(self) -> str:
        """Get the next question to ask"""
        missing_fields = self.get_missing_fields()
        if missing_fields:
            field = missing_fields[0]
            return self.questions.get(field, f"Please provide your {field.replace('_', ' ')}:")
        return None

    def process_message(self, user_message: str) -> str:
        """Process user message and return appropriate response"""
        try:
            self.conversation_history.append({"role": "user", "content": user_message})
            extracted_info = self.extract_information(user_message)
            
            # Validate and update extracted information
            valid_updates = {}
            for field, value in extracted_info.items():
                if hasattr(self.customer, field) and (value is not None):
                    if self.validate_field(field, value):
                        valid_updates[field] = value
            
            self.update_customer_info(valid_updates)
            self.update_required_fields()
            
            # Get next question or complete the phase
            next_question = self.get_next_question()
            if next_question:
                response = next_question
            else:
                if not self.check_phase_completion():
                    response = "Thank you for providing all the details! Moving to the next phase..."
                else:
                    response = self.handle_exit()
            
            self.conversation_history.append({"role": "assistant", "content": response})
            return response
            
        except Exception as e:
            error_msg = f"I apologize, but I encountered an error: {str(e)}"
            self.conversation_history.append({"role": "assistant", "content": error_msg})
            return error_msg

    def check_phase_completion(self):
        """Check if current phase is complete and move to next if needed"""
        missing_fields = self.get_missing_fields()
        
        if not missing_fields:
            next_phase = self._get_next_phase()
            if next_phase:
                self.current_collection_phase = next_phase
                print(f"✅ Moving to next phase: {self.current_collection_phase}")
                return False
            else:
                self.all_information_collected = True
                print("✅ All phases completed successfully.")
        return False
    
    def update_customer_info(self, updates: dict):
        """
        Update customer information with validated data.
        
        Args:
            updates (dict): Dictionary containing field names and values to update
        """
        try:
            for field, value in updates.items():
                if hasattr(self.customer, field):
                    # Convert string values for numeric fields
                    if field in ['max_monthly_payment', 'max_down_payment', 'min_budget', 'max_budget']:
                        try:
                            if isinstance(value, str):
                                # Remove currency symbols and commas
                                value = float(value.replace('$', '').replace(',', '').strip())
                        except ValueError:
                            continue
                    
                    # Update the field if validation passes
                    if self.validate_field(field, value):
                        setattr(self.customer, field, value)
                        print(f"✅ Updated {field}: {value}")
        except Exception as e:
            print(f"Error updating customer info: {str(e)}")

    def _get_next_phase(self):
        """Get the next phase in the collection process"""
        phases = list(self.required_fields.keys())
        current_index = phases.index(self.current_collection_phase)
        if current_index < len(phases) - 1:
            return phases[current_index + 1]
        return None

    def extract_information(self, text):
        """Extract information from user message using GPT"""
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
                - 'lease_down_payment': Only for lease transactions
                - 'max_budget': Overall maximum budget for purchase
                - 'max_down_payment': Down payment for purchase transactions

            3. Distinguishing Between `max_mileage` and `annual_mileage`:
                - `max_mileage`: Only for pre-owned cars
                - `annual_mileage`: Only for leasing

            4. When in transaction_details phase and transaction_type is 'lease':
                - Any mentioned payment amount should be interpreted appropriately
                - Do not update 'max_budget' or other payment fields
            
            5. Current transaction type is: {self.customer.transaction_type}
            
            6. Current collection phase is: {self.current_collection_phase}
            
            7. Ensure all fields match their validation rules:
                - car_condition: "new" or "pre-owned"
                - transaction_type: "buy" or "lease"
                - credit_rating: "Excellent", "Very Good", "Good", "Fair", "Poor"
                - car_type: "suv", "sedan", "truck", "van", "coupe"
                - year: Between 1900 and 2025
                - zip: 5 digits
            
            8. Handle trade-in responses appropriately:
                - Map "yes", "have trade-in", etc. to True
                - Map "no", "don't have trade-in", etc. to False

            Message: {text}

            Return a JSON object with only the newly extracted or updated information.
            """
            
            response = openai.ChatCompletion.create(
                model="gpt-4",
                temperature=0.7,
                messages=[
                    {"role": "system", "content": "You are a precise data extraction assistant. Extract only the specified information and return it as JSON."},
                    {"role": "user", "content": extraction_prompt}
                ]
            )

            response_content = response.choices[0].message.content.strip()
            print(f"Raw response content: {response_content}")
            
            # Parse the response as JSON
            response_content_str = response_content.replace("```json", "").replace("```", "").strip()
            extracted_info = json.loads(response_content_str)
            
            # Validate and clean the extracted information
            for field in list(extracted_info.keys()):
                value = extracted_info[field]
                
                # Remove invalid fields
                if not hasattr(self.customer, field):
                    extracted_info.pop(field)
                    continue
                
                # Handle validation for specific fields
                if field == 'zip' and not self.validation_rules['zip'](value):
                    extracted_info.pop(field)
                elif field == 'year' and not self.validation_rules['year'](value):
                    extracted_info.pop(field)
                elif field in ['car_condition', 'transaction_type', 'car_type', 'credit_rating']:
                    if field in self.validation_rules and not self.validation_rules[field](value):
                        extracted_info.pop(field)
                
                # Handle trade-in field
                if field == 'has_trade_in':
                    if isinstance(value, str):
                        if value.lower() in ['yes', 'true', 'y', '1', 'have']:
                            extracted_info['has_trade_in'] = True
                        elif value.lower() in ['no', 'false', 'n', '0', "don't", 'dont']:
                            extracted_info['has_trade_in'] = False
                        else:
                            extracted_info.pop(field)
            
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
            Give a detail full, it easy to understand and informative summary.
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
            # summary_file = f"conversation_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            # with open(summary_file, 'w', encoding='utf-8') as f:
            #     f.write(summary)
            return summary
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
            return summary
        else:
            summary = self.generate_summary()

        self.generate_summary_to_file()
        return """Thank you for your time. Here's a summary of our conversation:\n\n{summary}\n\nSummary file saved to 
        {summary_file}\n\nGoodbye!"""





def generate_customer_data(user_message, api_key=None, conversation_history=None):
    # Use default API key if none provided
    if not api_key:
        api_key = "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"

    # Initialize bot
    bot = CarSalesGPTBot(api_key)
 
    
    # Load previous conversation history if provided
    if conversation_history:
        bot.conversation_history = conversation_history

    try:
        # Process the message
        response = bot.process_message(user_message)
        
        # Check if conversation is complete
        is_complete = bot.all_information_collected
        
        
        # Get customer info and filter out empty fields
        customer_data = asdict(bot.customer)
        filled_fields = {
            key: value for key, value in customer_data.items() 
            if value is not None and value != "" and value != "N/A"
        }
       
        
        # Return the response and current state
        return {
            'response': response,
            'customer_info': filled_fields,
            'is_complete': is_complete,
            'current_phase': bot.current_collection_phase
        }

    except Exception as e:
        return {
            'error': str(e),
            'is_complete': False,
            'current_phase': bot.current_collection_phase
        }





def generate_conversation_summary(conversation_history=None, api_key=None):
    if not api_key:
        api_key ="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"
    
    bot = CarSalesGPTBot(api_key)
    bot.conversation_history = conversation_history
    
    summary = bot.generate_summary()
    return {
        'conversation_summary': summary if summary else "Conversation is not yet complete.",
        'is_complete': bot.all_information_collected
    }



if __name__ == "__main__": 
    api_key = "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"  # Replace with your actual OpenAI API key
    conversation_history = []  # Initialize an empty conversation history

    while True:
        try:
            user_input = input('You: ').strip()
            
            # Check for empty input
            if not user_input:
                print("Car Sales Assistant: I didn't catch that. Could you please repeat?")
                continue
            
            data = generate_customer_data(user_message=user_input, api_key=api_key, conversation_history=conversation_history)
            conversation_history.append({"role": "user", "content": user_input})
            conversation_history.append({"role": "assistant", "content": data['response']})
            print(f"Car Sales Assistant: {data['response']}")
        except Exception as e:
            print(f"An error occurred: {e}")







