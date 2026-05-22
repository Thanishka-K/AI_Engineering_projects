import streamlit as st
import pandas as pd
from PIL import Image
from google import genai
from google.genai import types
from pydantic import BaseModel
import os

st.set_page_config(page_title="AI Expense Vision", page_icon="🧾", layout="wide")
st.title("Multimodal Financial Expense Analyzer 🧾✨")

# Sidebar for Secure Configuration
api_key = st.sidebar.text_input("Enter Gemini API Key", type="password")

# Enforce Pydantic Structured Outputs to ensure data reliability
class ExpenseSchema(BaseModel):
    merchant_name: str
    transaction_date: str
    category: str
    total_amount: float
    detected_currency: str
    items_purchased: list[str]

if api_key:
    # Initialize the official Google GenAI Client
    client = genai.Client(api_key=api_key)

    # Initialize a dummy session state database to track historical uploads
    if "expense_db" not in st.session_state:
        st.session_state.expense_db = pd.DataFrame(columns=[
            "Merchant", "Date", "Category", "Amount", "Currency"
        ])

    uploaded_file = st.file_uploader("Upload a receipt or invoice image...", type=["jpg", "png", "jpeg"])

    if uploaded_file:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Invoice document", use_container_width=True)
            
        with col2:
            if st.button("Analyze with Gemini Vision"):
                with st.spinner("Processing unstructured document layout..."):
                    try:
                        # Call the multimodal model passing both the image and systemic extraction parameters
                        response = client.models.generate_content(
                            model='gemini-2.5-flash',
                            contents=[
                                image, 
                                "Analyze this financial receipt image. Extract data accurately matching the requested schema wrapper layout structure."
                            ],
                            config=types.GenerateContentConfig(
                                response_mime_type="application/json",
                                response_schema=ExpenseSchema,
                                temperature=0.1
                            ),
                        )
                        
                        # Parse structured JSON safely directly into dictionary object structures
                        import json
                        parsed_data = json.loads(response.text)
                        
                        st.subheader("📊 Extracted Ledger Details")
                        st.json(parsed_data)
                        
                        # Append directly into our runtime dataframe
                        new_row = {
                            "Merchant": parsed_data["merchant_name"],
                            "Date": parsed_data["transaction_date"],
                            "Category": parsed_data["category"],
                            "Amount": parsed_data["total_amount"],
                            "Currency": parsed_data["detected_currency"]
                        }
                        st.session_state.expense_db = pd.concat([
                            st.session_state.expense_db, 
                            pd.DataFrame([new_row])
                        ], ignore_index=True)
                        
                    except Exception as e:
                        st.error(f"Execution Error encountered: {str(e)}")

    # Dashboard Metrics Section
    if not st.session_state.expense_db.empty:
        st.write("---")
        st.subheader("📈 Core Spend Analytics Ledger Summary")
        st.dataframe(st.session_state.expense_db, use_container_width=True)
        
        # Display automated aggregations
        total_spent = st.session_state.expense_db["Amount"].sum()
        st.metric(label="Aggregated Running Expenditures (Local Baseline)", value=f"{total_spent:.2f}")

else:
    st.info("Please keyspace your Google API access parameter configurations inside the left sidebar panel interface context.")
