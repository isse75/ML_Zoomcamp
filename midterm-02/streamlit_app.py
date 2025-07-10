import streamlit as st
import requests
import json

# Configure page
st.set_page_config(
    page_title="Bank Deposit Prediction",
    page_icon="🏦",
    layout="wide"
)

# Title and description
st.title("🏦 Bank Deposit Prediction")
st.write("Predict whether a customer will make a term deposit based on their profile")

# API endpoint
API_URL = "http://34.248.88.252:9696/predict"

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.header("Customer Information")
    
    age = st.number_input("Age", min_value=18, max_value=100, value=35)
    
    job = st.selectbox("Job", [
        "admin.", "blue-collar", "entrepreneur", "housemaid",
        "management", "retired", "self-employed", "services",
        "student", "technician", "unemployed", "unknown"
    ])
    
    marital = st.selectbox("Marital Status", [
        "divorced", "married", "single"
    ])
    
    education = st.selectbox("Education", [
        "primary", "secondary", "tertiary", "unknown"
    ])
    
    default = st.selectbox("Has Credit in Default?", ["yes", "no"])
    
    previous = st.number_input("Number of Previous Contacts", min_value=0, value=0)

with col2:
    st.header("Contact Information")
    
    housing = st.selectbox("Housing Loan?", ["yes", "no"])
    loan = st.selectbox("Personal Loan?", ["yes", "no"])
    contact = st.selectbox("Contact Type", ["cellular", "telephone", "unknown"])
    
    month = st.selectbox("Last Contact Month", [
        "jan", "feb", "mar", "apr", "may", "jun",
        "jul", "aug", "sep", "oct", "nov", "dec"
    ])
    
    day_of_week = st.selectbox("Last Contact Day of Week", [
        "mon", "tue", "wed", "thu", "fri"
    ])
    
    poutcome = st.selectbox("Previous Campaign Outcome", [
        "failure", "nonexistent", "success"
    ])
    
    campaign = st.number_input("Number of Contacts This Campaign", min_value=1, value=2)

# Prediction button
if st.button("🔮 Predict Deposit Decision", type="primary"):
    # Prepare data for API - EXACTLY match training features
    customer_data = {
        "age": age,
        "job": job,
        "marital": marital,
        "education": education,
        "default": default,
        "housing": housing,
        "loan": loan,
        "contact": contact,
        "month": month,
        "day_of_week": day_of_week,
        "campaign": campaign,
        "previous": previous,
        "poutcome": poutcome
    }
    
    try:
        # Make API call
        with st.spinner("Making prediction..."):
            response = requests.post(API_URL, json=customer_data)
            
        if response.status_code == 200:
            result = response.json()
            
            # Display results
            st.success("Prediction Complete!")
            
            # Create metrics display
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric(
                    label="Deposit Probability", 
                    value=f"{result['deposit_probability']:.1%}"
                )
            
            with col2:
                decision = "Will Subscribe" if result['will_deposit'] else "Will Not Subscribe"
                st.metric(label="Decision", value=decision)
            
            # Recommendation
            if result['will_deposit']:
                st.balloons()
                st.success("✅ **Recommendation**: Contact this customer with a term deposit offer!")
            else:
                st.warning("❌ **Recommendation**: Customer unlikely to subscribe. Consider other products.")
                
            # Show raw data
            with st.expander("View Raw Prediction Data"):
                st.json(result)
                
        else:
            st.error(f"API Error: {response.status_code}")
            st.write("Response:", response.text)
            
    except requests.exceptions.ConnectionError:
        st.error("❌ **Connection Error**: Cannot connect to the prediction API.")
        st.write("Make sure your FastAPI service is running on EC2.")
        st.code(f"API URL: {API_URL}")
        
    except Exception as e:
        st.error(f"❌ **Error**: {str(e)}")

# Sidebar with info
st.sidebar.header("About This App")
st.sidebar.write("""
This app predicts whether a bank customer will subscribe to a term deposit based on their profile and previous marketing interactions.

**Features Used:**
- Demographics (age, job, education)
- Financial info (balance, loans)
- Contact information
- Campaign history

**Model:** XGBoost Classifier deployed on AWS EC2
""")

st.sidebar.header("API Status")
try:
    health_response = requests.get("http://34.248.88.252:9696/")
    if health_response.status_code == 200:
        st.sidebar.success("✅ API Online")
    else:
        st.sidebar.error("❌ API Issues")
except:
    st.sidebar.error("❌ API Offline")
