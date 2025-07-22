import React, { useState } from 'react';
import axios from 'axios';
import './App.css'; // You can add basic styling here

function App() {
  // State for form inputs
  const [formData, setFormData] = useState({
    Pregnancies: '1',
    Glucose: '120',
    BloodPressure: '70',
    SkinThickness: '20',
    Insulin: '80',
    BMI: '30.5',
    DiabetesPedigreeFunction: '0.47',
    Age: '35',
  });
  
  // State for the prediction result
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');

  const handleChange = (e) => {
    setFormData({
      ...formData,
      [e.target.name]: e.target.value,
    });
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);

    // Convert form data values to numbers
    const dataToSubmit = Object.fromEntries(
      Object.entries(formData).map(([key, value]) => [key, parseFloat(value)])
    );

    try {
      // Make a POST request to the Flask backend
      const response = await axios.post('http://127.0.0.1:5001/predict', dataToSubmit);
      setResult(response.data);
    } catch (err) {
      setError('An error occurred. Please check the input and try again.');
      console.error(err);
    }
  };

  return (
    <div className="container">
      <h1>🏥 AI Health Risk Predictor</h1>
      <p>Enter your health metrics to predict the risk of diabetes. This is a demo and not a substitute for professional medical advice.</p>
      
      <form onSubmit={handleSubmit} className="health-form">
        {Object.keys(formData).map((key) => (
          <div className="form-group" key={key}>
            <label>{key.replace(/([A-Z])/g, ' $1').trim()}</label>
            <input
              type="number"
              name={key}
              value={formData[key]}
              onChange={handleChange}
              required
              step="any"
            />
          </div>
        ))}
        <button type="submit">Predict Risk</button>
      </form>

      {result && (
        <div className="result-card">
          <h2>Prediction Result</h2>
          <p className={result.prediction === 1 ? 'risk' : 'no-risk'}>
            {result.prediction === 1 ? 'High Risk of Diabetes Detected' : 'Low Risk of Diabetes Detected'}
          </p>
          <p>Confidence Score: {result.confidence_score}</p>
        </div>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default App;