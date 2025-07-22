from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
from dotenv import load_dotenv

# --- New Imports for LangChain ---
from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables from .env file (for OPENAI_API_KEY)
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)
CORS(app) # Enable Cross-Origin Resource Sharing

# --- Existing Model Loading Logic (No Changes) ---
try:
    model = joblib.load('../ml_model/diabetes_model.joblib')
    scaler = joblib.load('../ml_model/scaler.joblib')
    model_columns = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")
    model = None
    scaler = None

# --- New LangChain Agent Setup ---

# This tool allows the AI agent to use your prediction model.
@tool
def get_health_risk_prediction(
    Pregnancies: int, Glucose: int, BloodPressure: int, 
    SkinThickness: int, Insulin: int, BMI: float, 
    DiabetesPedigreeFunction: float, Age: int
) -> str:
    """
    Predicts the risk of diabetes based on a user's health metrics. 
    Use this tool when a user asks for a health prediction or risk assessment and has provided all necessary parameters.
    """
    if not model or not scaler:
        return "Sorry, the prediction model is not available at the moment."
    
    input_data = pd.DataFrame([[
        Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age
    ]], columns=model_columns)
    
    scaled_data = scaler.transform(input_data)
    prediction = model.predict(scaled_data)[0]
    confidence = model.predict_proba(scaled_data)[0].max()

    if prediction == 1:
        return f"The model predicts a HIGH RISK of diabetes with a confidence of {confidence*100:.2f}%. The user should be advised to consult a doctor."
    else:
        return f"The model predicts a LOW RISK of diabetes with a confidence of {confidence*100:.2f}%. The user should still be encouraged to maintain a healthy lifestyle."

# Set up the Large Language Model (LLM) and tools
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [get_health_risk_prediction]

# Create the prompt template that defines the agent's behavior
prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful and empathetic AI healthcare assistant. Your role is to provide supportive information, and if a user provides all necessary health metrics (Pregnancies, Glucose, etc.), you can use the health risk prediction tool. NEVER give medical advice. Always suggest consulting a professional doctor for any health concerns."),
    MessagesPlaceholder(variable_name="chat_history", optional=True),
    ("user", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad"),
])

# Create the agent and the object that runs it (the "executor")
agent = create_openai_tools_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

# Simple in-memory store for chat history
chat_history = []

# --- Existing Endpoints (No Changes) ---
@app.route('/')
def home():
    return "Healthcare AI Assistant Backend is running!"

@app.route('/predict', methods=['POST'])
def predict():
    if not model or not scaler:
        return jsonify({'error': 'Model not loaded, check server logs'}), 500

    json_data = request.get_json()
    
    try:
        data = pd.DataFrame([json_data], columns=model_columns)
    except Exception as e:
        return jsonify({'error': f'Invalid input data: {e}'}), 400

    scaled_data = scaler.transform(data)
    prediction = model.predict(scaled_data)
    prediction_proba = model.predict_proba(scaled_data)
    
    return jsonify({
        'prediction': int(prediction[0]),
        'confidence_score': f"{np.max(prediction_proba[0]) * 100:.2f}%"
    })

# --- New Chat Endpoint ---
@app.route('/chat', methods=['POST'])
def chat():
    global chat_history
    user_message = request.get_json().get('message')
    if not user_message:
        return jsonify({"error": "No message provided"}), 400

    # Get the response from the AI agent
    response = agent_executor.invoke({
        "input": user_message,
        "chat_history": chat_history
    })
    
    # Add the current conversation to history
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": response['output']})

    # Optional: limit history size to save memory/tokens
    if len(chat_history) > 10:
        chat_history = chat_history[-10:]
        
    return jsonify({"reply": response['output']})

# --- Main Execution Block (No Changes) ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)