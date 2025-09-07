import React, { useState } from 'react';
import axios from 'axios';
import './App.css';

const HEART_FIELDS = [
  { name: 'Age', type: 'number' },
  { name: 'Sex', type: 'select', options: ['M', 'F'] },
  { name: 'ChestPainType', type: 'select', options: ['ATA', 'NAP', 'ASY', 'TA'] },
  { name: 'RestingBP', type: 'number' },
  { name: 'Cholesterol', type: 'number' },
  { name: 'FastingBS', type: 'number' },
  { name: 'RestingECG', type: 'select', options: ['Normal', 'ST', 'LVH'] },
  { name: 'MaxHR', type: 'number' },
  { name: 'ExerciseAngina', type: 'select', options: ['Y', 'N'] },
  { name: 'Oldpeak', type: 'number' },
  { name: 'ST_Slope', type: 'select', options: ['Up', 'Flat', 'Down'] },
];

// Common BRFSS-style fields (subset) for diabetes/hypertension forms
const BRFSS_COMMON_FIELDS = [
  { name: 'BMI', type: 'number' },
  { name: 'HighBP', type: 'number' },
  { name: 'HighChol', type: 'number' },
  { name: 'CholCheck', type: 'number' },
  { name: 'Smoker', type: 'number' },
  { name: 'Stroke', type: 'number' },
  { name: 'HeartDiseaseorAttack', type: 'number' },
  { name: 'PhysActivity', type: 'number' },
  { name: 'Fruits', type: 'number' },
  { name: 'Veggies', type: 'number' },
  { name: 'HvyAlcoholConsump', type: 'number' },
  { name: 'AnyHealthcare', type: 'number' },
  { name: 'NoDocbcCost', type: 'number' },
  { name: 'GenHlth', type: 'number' },
  { name: 'MentHlth', type: 'number' },
  { name: 'PhysHlth', type: 'number' },
  { name: 'DiffWalk', type: 'number' },
  { name: 'Sex', type: 'number' },
  { name: 'Age', type: 'number' },
  { name: 'Education', type: 'number' },
  { name: 'Income', type: 'number' },
];

const FORM_FIELDS = {
  diabetes: BRFSS_COMMON_FIELDS,
  heart: HEART_FIELDS,
  hypertension: BRFSS_COMMON_FIELDS,
};

const DEFAULTS = (fields) =>
  fields.reduce((acc, f) => {
    acc[f.name] = f.type === 'number' ? '' : (f.options ? f.options[0] : '');
    return acc;
  }, {});

function App() {
  const [condition, setCondition] = useState('diabetes');
  const [formData, setFormData] = useState(DEFAULTS(FORM_FIELDS['diabetes']));
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [explain, setExplain] = useState(null);

  const handleConditionChange = (e) => {
    const value = e.target.value;
    setCondition(value);
    setFormData(DEFAULTS(FORM_FIELDS[value]));
    setResult(null);
    setError('');
  };

  const handleChange = (e) => {
    setFormData({ ...formData, [e.target.name]: e.target.value });
  };

  const normalizePayload = (obj, fields) => {
    const out = {};
    fields.forEach((f) => {
      const raw = obj[f.name];
      if (f.type === 'number') {
        const num = raw === '' || raw === undefined ? null : Number(raw);
        out[f.name] = Number.isFinite(num) ? num : null;
      } else {
        out[f.name] = raw;
      }
    });
    return out;
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setResult(null);
    setExplain(null);
    try {
      const payload = {
        predictionType: condition,
        ...normalizePayload(formData, FORM_FIELDS[condition]),
      };
      const response = await axios.post('http://127.0.0.1:5001/predict', payload);
      setResult(response.data);
    } catch (err) {
      setError('An error occurred. Please check the input and try again.');
      console.error(err);
    }
  };

  const handleExplain = async () => {
    try {
      const payload = {
        predictionType: condition,
        ...normalizePayload(formData, FORM_FIELDS[condition]),
      };
      const response = await axios.post('http://127.0.0.1:5001/explain', payload);
      setExplain(response.data);
    } catch (err) {
      setError('Failed to generate explanation.');
      console.error(err);
    }
  };

  const fields = FORM_FIELDS[condition];

  return (
    <div className="container">
      <h1 style={{display:'flex',alignItems:'center',gap:12}}>
        <img src="/health-icon.svg" alt="Health" width="32" height="32" />
        AI Health Risk Predictor
      </h1>
      <p>Select a condition, enter your metrics, and submit for a prediction.</p>

      <div className="selector-row">
        <label htmlFor="condition">Condition</label>
        <select id="condition" value={condition} onChange={handleConditionChange}>
          <option value="diabetes">Diabetes</option>
          <option value="heart">Heart Disease</option>
          <option value="hypertension">Hypertension</option>
        </select>
      </div>

      <form onSubmit={handleSubmit} className="health-form card">
        <h2 className="card-title">
          {condition === 'diabetes' && 'Diabetes Inputs'}
          {condition === 'heart' && 'Heart Disease Inputs'}
          {condition === 'hypertension' && 'Hypertension Inputs'}
        </h2>

        <div className="grid">
          {fields.map((f) => (
            <div className="form-group" key={f.name}>
              <label>{f.name.replace(/([A-Z])/g, ' $1').trim()}</label>
              {f.type === 'select' ? (
                <select name={f.name} value={formData[f.name]} onChange={handleChange}>
                  {f.options.map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              ) : (
                <input
                  type="number"
                  name={f.name}
                  value={formData[f.name]}
                  onChange={handleChange}
                  step="any"
                  required
                />
              )}
            </div>
          ))}
        </div>

        <button type="submit" className="primary">Predict Risk</button>
      </form>

      {result && (
        <div className="result-card card">
          <h2 className="card-title">Prediction Result</h2>
          <div className="result-row">
            <span className="badge">{result.predictionType}</span>
            <span className={result.prediction === 1 ? 'risk' : 'no-risk'}>
              {result.prediction === 1 ? 'High Risk Detected' : 'Low Risk Detected'}
            </span>
          </div>
          {result.confidence_score && (
            <p>Confidence: <strong>{result.confidence_score}</strong></p>
          )}
          <button className="secondary" onClick={handleExplain}>Explain Prediction</button>
          {explain && explain.details && (
            <div className="card xai-section">
              <h3 className="card-title xai-title">Top Influential Features</h3>
              {(() => {
                const items = explain.details;
                const sorted = [...items].sort((a,b) => Math.abs(b.shap) - Math.abs(a.shap));
                const positives = sorted.filter(x => x.shap > 0).slice(0,3);
                const negatives = sorted.filter(x => x.shap < 0).slice(0,3);
                return (
                  <div className="xai-grid">
                    <div>
                      <h4>Top Positive</h4>
                      <ul>
                        {positives.map(p => (
                          <li key={p.feature}><strong>{p.feature}</strong>: {String(p.value)} (SHAP {p.shap.toFixed(4)})</li>
                        ))}
                      </ul>
                    </div>
                    <div>
                      <h4>Top Negative</h4>
                      <ul>
                        {negatives.map(n => (
                          <li key={n.feature}><strong>{n.feature}</strong>: {String(n.value)} (SHAP {n.shap.toFixed(4)})</li>
                        ))}
                      </ul>
                    </div>
                  </div>
                );
              })()}
            </div>
          )}
        </div>
      )}
      {error && <p className="error">{error}</p>}
    </div>
  );
}

export default App;