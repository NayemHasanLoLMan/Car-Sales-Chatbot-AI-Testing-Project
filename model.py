from typing import List, Optional
from pydantic import BaseModel, EmailStr, Field
from datetime import datetime, timezone
from django.db import models



# Pydantic Models
class CustomerInfo(BaseModel):
    first_name: str
    last_name: str
    email: Optional[EmailStr] = None
    phone: Optional[str] = None
    zip: str
    car_condition: str
    transaction_type: str
    payment_method: str
    credit_rating: str
    max_monthly_payment: Optional[float] = None
    max_down_payment: Optional[float] = None
    max_finance_term: Optional[int] = None
    lease_term: Optional[str] = None
    annual_mileage: Optional[str] = None
    lease_down_payment: Optional[str] = None
    car_type: str
    make: str
    year: Optional[int] = None
    model: Optional[str] = None
    max_mileage: Optional[int] = None
    desired_features: Optional[dict] = None
    has_trade_in: bool
    min_budget: Optional[float] = None
    max_budget: float

class ConversationMessage(BaseModel):
    role: str
    content: str

class CarPurchaseConversation(BaseModel):
    customer_info: CustomerInfo
    conversation_history: List[ConversationMessage]
    timestamp: datetime
    collection_phase: str
    all_information_collected: bool

# Example usage:
conversation_data = CarPurchaseConversation(
    customer_info=CustomerInfo(
        first_name="Hasan",
        last_name="Mahmood",
        email="hashr q2@f 3w",
        phone=None,
        zip="01245",
        car_condition="new",
        transaction_type="buy",
        payment_method="cash",
        credit_rating="Very Good",
        max_monthly_payment=None,
        max_down_payment=3000,
        max_finance_term=32,
        lease_term="N/A",
        annual_mileage="N/A",
        lease_down_payment="N/A",
        car_type="suv",
        make="bmw",
        year=None,
        model=None,
        max_mileage=None,
        desired_features=None,
        has_trade_in=False,
        min_budget=None,
        max_budget=53000.0
    ),
    conversation_history=[
        ConversationMessage(role="user", content="my name is hasan"),
        ConversationMessage(role="assistant", content="Thank you, Hasan. May I have your last name, please?"),
        # Additional messages...
    ],
        conversation_summery= models.CharField(max_length=1000) ,
    current_time = datetime.now(timezone.utc),
    collection_phase="personal_info",
    all_information_collected=False
)
