from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import pandas as pd
import shap
from dotenv import load_dotenv
from pathlib import Path

# --- New Imports for LangChain ---
from langchain_openai import ChatOpenAI
from langchain.agents import tool, AgentExecutor, create_openai_tools_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Load environment variables from .env file (for OPENAI_API_KEY)
load_dotenv()

# Initialize the Flask application
app = Flask(__name__)
CORS(app)  # Enable Cross-Origin Resource Sharing

ML_DIR = (Path(__file__).resolve().parent.parent / 'ml_model').resolve()


def _load_pipeline_and_scaler(prefix: str):
    pipeline_path = ML_DIR / f"{prefix}_model.joblib"
    scaler_path = ML_DIR / f"{prefix}_scaler.joblib"
    pipeline = joblib.load(pipeline_path)
    scaler = None
    try:
        scaler = joblib.load(scaler_path)
    except Exception:
        scaler = None

    # Derive expected base feature columns from the pipeline's preprocessor
    expected_columns = []
    try:
        pre = pipeline.named_steps.get('preprocess')
        if pre is not None and hasattr(pre, 'transformers_'):
            for name, trans, cols in pre.transformers_:
                if isinstance(cols, list):
                    expected_columns.extend(cols)
    except Exception:
        expected_columns = []

    return pipeline, scaler, expected_columns


# Load all disease-specific models/scalers
models = {}
try:
    models['diabetes'] = _load_pipeline_and_scaler('diabetes')
    models['heart'] = _load_pipeline_and_scaler('heart')
    models['hypertension'] = _load_pipeline_and_scaler('hypertension')
    print("All models loaded successfully from:", ML_DIR)
except Exception as e:
    print(f"Error loading models: {e}")

# --- New LangChain Agent Setup ---


def _run_structured_prediction(prediction_type: str, payload: dict) -> str:
    if prediction_type not in models:
        return f"Unknown prediction type '{prediction_type}'."
    pipeline, _scaler, expected_cols = models[prediction_type]
    if pipeline is None:
        return f"Model for '{prediction_type}' is not available."

    df = pd.DataFrame([payload])
    if expected_cols:
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]

    try:
        proba = pipeline.predict_proba(df) if hasattr(
            pipeline, 'predict_proba') else None
        pred = pipeline.predict(df)[0]
        conf = float(np.max(proba[0])) if proba is not None else None
    except Exception as e:
        return f"Prediction failed: {e}"

    label = 'HIGH RISK' if int(pred) == 1 else 'LOW RISK'
    conf_txt = f" with a confidence of {conf*100:.2f}%" if conf is not None else ""
    return f"The model predicts {label}{conf_txt}. Always consult a medical professional for guidance."


# Renamed: diabetes risk tool (BRFSS-style inputs)
@tool
def get_diabetes_risk_prediction(
    BMI: float,
    HighBP: int,
    HighChol: int,
    Smoker: int,
    PhysActivity: int,
    GenHlth: int,
    Age: int,
    Sex: int,
    CholCheck: int = 0,
    HeartDiseaseorAttack: int = 0,
    HvyAlcoholConsump: int = 0,
    AnyHealthcare: int = 1,
    NoDocbcCost: int = 0,
    MentHlth: int = 0,
    PhysHlth: int = 0,
    DiffWalk: int = 0,
    Education: int = 3,
    Income: int = 3,
) -> str:
    """Predicts diabetes risk using BRFSS-style inputs (e.g., BMI, HighBP, HighChol, Smoker, PhysActivity, GenHlth, Age, Sex, etc.)."""
    payload = locals().copy()
    return _run_structured_prediction('diabetes', payload)


# Heart disease risk tool (UCI Heart-style inputs)
@tool
def get_heart_disease_risk_prediction(
    Age: int,
    Sex: str,
    ChestPainType: str,
    RestingBP: float,
    Cholesterol: float,
    FastingBS: int,
    RestingECG: str,
    MaxHR: float,
    ExerciseAngina: str,
    Oldpeak: float,
    ST_Slope: str,
) -> str:
    """Predicts heart disease risk using UCI Heart inputs (Age, Sex[M/F], ChestPainType[ATA/NAP/ASY/TA], RestingBP, Cholesterol, FastingBS, RestingECG[Normal/ST/LVH], MaxHR, ExerciseAngina[Y/N], Oldpeak, ST_Slope[Up/Flat/Down])."""
    payload = locals().copy()
    return _run_structured_prediction('heart', payload)


# Hypertension risk tool (BRFSS-style inputs similar to diabetes)
@tool
def get_hypertension_risk_prediction(
    BMI: float,
    HighChol: int,
    Smoker: int,
    PhysActivity: int,
    GenHlth: int,
    Age: int,
    Sex: int,
    CholCheck: int = 0,
    HeartDiseaseorAttack: int = 0,
    HvyAlcoholConsump: int = 0,
    AnyHealthcare: int = 1,
    NoDocbcCost: int = 0,
    MentHlth: int = 0,
    PhysHlth: int = 0,
    DiffWalk: int = 0,
    Education: int = 3,
    Income: int = 3,
) -> str:
    """Predicts hypertension risk using BRFSS-style inputs (BMI, HighChol, lifestyle flags, Age, Sex, GenHlth, etc.)."""
    payload = locals().copy()
    return _run_structured_prediction('hypertension', payload)


# Set up the Large Language Model (LLM) and tools
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
tools = [
    get_diabetes_risk_prediction,
    get_heart_disease_risk_prediction,
    get_hypertension_risk_prediction,
]

# Create the prompt template that defines the agent's behavior
prompt = ChatPromptTemplate.from_messages([
    ("system", (
        "You are a helpful and empathetic AI healthcare assistant. You can estimate risk using tools for: "
        "1) Diabetes (BRFSS-style inputs like BMI, HighBP, HighChol, Smoker, PhysActivity, GenHlth, Age, Sex, etc.), "
        "2) Heart Disease (UCI Heart inputs: Age, Sex[M/F], ChestPainType, RestingBP, Cholesterol, FastingBS, RestingECG, MaxHR, ExerciseAngina[Y/N], Oldpeak, ST_Slope), and "
        "3) Hypertension (BRFSS-style inputs similar to diabetes). "
        "If the user asks e.g. 'check my risk for heart disease', gather the required parameters, then call the appropriate tool. "
        "NEVER give medical advice. Encourage consulting a professional doctor for any health concerns."
    )),
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
    payload = request.get_json() or {}
    prediction_type = payload.pop('predictionType', None)
    if prediction_type is None:
        return jsonify({'error': "'predictionType' is required ('diabetes' | 'heart' | 'hypertension')"}), 400

    if prediction_type not in models:
        return jsonify({'error': f"Unsupported predictionType '{prediction_type}'."}), 400

    pipeline, scaler, expected_columns = models[prediction_type]
    if pipeline is None:
        return jsonify({'error': f"Model for '{prediction_type}' is not loaded."}), 500

    # Prepare single-row DataFrame from remaining fields
    input_df = pd.DataFrame([payload])

    # Align to expected columns if available
    if expected_columns:
        for col in expected_columns:
            if col not in input_df.columns:
                input_df[col] = 0
        # Keep only expected columns; extra fields are dropped
        input_df = input_df[expected_columns]

    # Predict using the full pipeline (includes scaler and encoders)
    try:
        proba = None
        if hasattr(pipeline, 'predict_proba'):
            proba = pipeline.predict_proba(input_df)
        pred = pipeline.predict(input_df)
    except Exception as e:
        return jsonify({'error': f'Prediction failed: {e}'}), 400

    confidence = None
    if proba is not None:
        confidence = float(np.max(proba[0]))

    return jsonify({
        'predictionType': prediction_type,
        'prediction': int(pred[0]) if hasattr(pred, '__len__') else int(pred),
        'confidence_score': f"{confidence * 100:.2f}%" if confidence is not None else None
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


# --- New Explainability Endpoint (XAI via SHAP) ---


DATA_FILES = {
    'diabetes': 'diabetes_012_health_indicators_BRFSS2015.csv',
    'heart': 'heart.csv',
    'hypertension': 'hypertension_dataset.csv',
}

TARGET_COLUMNS = {
    'diabetes': 'Diabetes_012',
    'heart': 'HeartDisease',
    'hypertension': 'Hypertension',
}


def _prepare_input_df(prediction_type: str, payload: dict, expected_columns: list[str]) -> pd.DataFrame:
    df = pd.DataFrame([payload])
    if expected_columns:
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_columns]
    return df


def _load_background_df(prediction_type: str, expected_columns: list[str], sample_size: int = 100) -> pd.DataFrame:
    csv_name = DATA_FILES.get(prediction_type)
    if csv_name is None:
        raise ValueError(
            f"No data file mapping for prediction type '{prediction_type}'.")
    csv_path = ML_DIR / csv_name
    if not csv_path.exists():
        raise FileNotFoundError(f"Background data not found at {csv_path}")
    df = pd.read_csv(csv_path)
    target_col = TARGET_COLUMNS.get(prediction_type)
    if target_col in df.columns:
        df = df.drop(columns=[target_col])
    # Align to expected columns if provided
    if expected_columns:
        for col in expected_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_columns]
    # Sample a manageable background for Kernel SHAP
    if len(df) > sample_size:
        df = df.sample(n=sample_size, random_state=42)
    return df.reset_index(drop=True)


@app.route('/explain', methods=['POST'])
def explain():
    payload = request.get_json() or {}
    prediction_type = payload.pop('predictionType', None)
    if prediction_type is None:
        return jsonify({'error': "'predictionType' is required ('diabetes' | 'heart' | 'hypertension')"}), 400

    if prediction_type not in models:
        return jsonify({'error': f"Unsupported predictionType '{prediction_type}'."}), 400

    pipeline, _scaler, expected_columns = models[prediction_type]
    if pipeline is None:
        return jsonify({'error': f"Model for '{prediction_type}' is not loaded."}), 500

    try:
        input_df = _prepare_input_df(
            prediction_type, payload, expected_columns)
        background_df = _load_background_df(
            prediction_type, expected_columns, sample_size=100)

        # Define a probability function for class 1
        def predict_proba_one(X: pd.DataFrame | np.ndarray) -> np.ndarray:
            X_df = pd.DataFrame(X, columns=input_df.columns) if not isinstance(
                X, pd.DataFrame) else X
            probs = pipeline.predict_proba(X_df)
            # Return probability of the positive class
            return probs[:, 1]

        explainer = shap.KernelExplainer(predict_proba_one, background_df)
        shap_values = explainer.shap_values(input_df, nsamples='auto')
        base_value = float(explainer.expected_value)

        # Build feature-wise attributions
        shap_list = []
        instance_values = input_df.iloc[0].to_dict()
        # shap_values can be shape (n_features,) for single instance
        sv = np.array(shap_values).reshape(-1)
        for i, col in enumerate(input_df.columns):
            shap_list.append({
                'feature': col,
                'value': instance_values[col],
                'shap': float(sv[i])
            })

        return jsonify({
            'predictionType': prediction_type,
            'base_value': base_value,
            'details': shap_list
        })
    except Exception as e:
        return jsonify({'error': f'Explanation failed: {e}'}), 400


# --- Main Execution Block (No Changes) ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)
