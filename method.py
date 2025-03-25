import json
from personalinfo import CustomerInfo
from dataclasses import asdict
{
  "conversation_history": [
    {
      "role": "user",
      "content": "my name is hasan"
    },
    {
      "role": "assistant",
      "content": "Thank you, Hasan. May I have your last name, please?"
    },
    {
      "role": "user",
      "content": "mahmood"
    },
    {
      "role": "assistant",
      "content": "Great, Hasan Mahmood. Could you please provide me with your email address?"
    },
    {
      "role": "user",
      "content": "hashr q2@f 3w"
    }
  ]
}


def customer_data(conversation_history = historyJson):
    data =CustomerInfo.first_name
    # print(data)

    return  { 'customer_info': asdict(self.customer)}


# Function to Load JSON and Save to Django DB
def save_conversation_to_db(json_data):
    data = json.loads(json_data)
    
    customer_info = CustomerInfoModel.objects.create(
        first_name=data["customer_info"]["first_name"],
        last_name=data["customer_info"]["last_name"],
        email=data["customer_info"].get("email"),
        phone=data["customer_info"].get("phone"),
        zip=data["customer_info"]["zip"],
        car_condition=data["customer_info"]["car_condition"],
        transaction_type=data["customer_info"]["transaction_type"],
        payment_method=data["customer_info"]["payment_method"],
        credit_rating=data["customer_info"]["credit_rating"],
        max_monthly_payment=data["customer_info"].get("max_monthly_payment"),
        max_down_payment=data["customer_info"].get("max_down_payment"),
        max_finance_term=data["customer_info"].get("max_finance_term"),
        lease_term=data["customer_info"].get("lease_term"),
        annual_mileage=data["customer_info"].get("annual_mileage"),
        lease_down_payment=data["customer_info"].get("lease_down_payment"),
        car_type=data["customer_info"]["car_type"],
        make=data["customer_info"]["make"],
        year=data["customer_info"].get("year"),
        model=data["customer_info"].get("model"),
        max_mileage=data["customer_info"].get("max_mileage"),
        desired_features=data["customer_info"].get("desired_features"),
        has_trade_in=data["customer_info"]["has_trade_in"],
        min_budget=data["customer_info"].get("min_budget"),
        max_budget=data["customer_info"]["max_budget"]
    )
    
    conversation = CarPurchaseConversationModel.objects.create(
        customer_info=customer_info,
        timestamp=data["timestamp"],
        collection_phase=data["collection_phase"],
        all_information_collected=data["all_information_collected"]
    )
    
    for msg in data["conversation_history"]:
        ConversationMessageModel.objects.create(


            role=msg["role"],
            content=msg["content"],
            conversation=conversation
        )
    
    return conversation
