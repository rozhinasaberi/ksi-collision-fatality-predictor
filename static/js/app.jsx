const { useState } = React;

const APP_CONFIG = window.APP_CONFIG || {};
const API_BASE = APP_CONFIG.apiBase || "";

const FIELD_META = {
  ROAD_CLASS: {
    label: "Road Class",
  },
  DISTRICT: {
    label: "District",
  },
  ACCLOC: {
    label: "Collision Location",
  },
  TRAFFCTL: {
    label: "Traffic Control",
  },
  VISIBILITY: {
    label: "Visibility",
  },
  LIGHT: {
    label: "Lighting",
  },
  RDSFCOND: {
    label: "Road Surface Condition",
  },
  IMPACTYPE: {
    label: "Impact Type",
  },
  HOUR: {
    label: "Hour of Collision",
  },
  PEDESTRIAN: {
    label: "Pedestrian Involved",
  },
  CYCLIST: {
    label: "Cyclist Involved",
  },
  AUTOMOBILE: {
    label: "Automobile Involved",
  },
  MOTORCYCLE: {
    label: "Motorcycle Involved",
  },
  TRUCK: {
    label: "Truck Involved",
  },
  PASSENGER: {
    label: "Passenger Present",
  },
  SPEEDING: {
    label: "Speeding Reported",
  },
  AG_DRIV: {
    label: "Aggressive Driving",
  },
  REDLIGHT: {
    label: "Red Light Violation",
  },
  ALCOHOL: {
    label: "Alcohol Involved",
  },
  DISABILITY: {
    label: "Medical/Physical Disability",
  },
};

const SCENARIO_FIELDS = [
  "ROAD_CLASS",
  "DISTRICT",
  "ACCLOC",
  "TRAFFCTL",
  "VISIBILITY",
  "LIGHT",
  "RDSFCOND",
  "IMPACTYPE",
];

const RISK_FLAGS = APP_CONFIG.flagColumns || [];

function buildInitialFormState() {
  const formState = { HOUR: 14 };
  SCENARIO_FIELDS.forEach((field) => {
    const options = APP_CONFIG.options?.[field] || [];
    formState[field] = options[0] || "";
  });
  RISK_FLAGS.forEach((field) => {
    formState[field] = false;
  });
  return formState;
}

function modelCardSummary(label) {
  if (label.includes("Decision Tree")) {
    return "Rule-based model that is straightforward to explain during the class demo.";
  }
  if (label.includes("Random Forest")) {
    return "Ensemble tree model that combines many trees for a more robust prediction.";
  }
  return "Trained classifier available for local inference through the Flask API.";
}

function validateForm(form) {
  const nextErrors = {};
  if (form.HOUR === "" || Number.isNaN(Number(form.HOUR))) {
    nextErrors.HOUR = "Enter an hour between 0 and 23.";
  } else {
    const hour = Number(form.HOUR);
    if (!Number.isInteger(hour) || hour < 0 || hour > 23) {
      nextErrors.HOUR = "Hour must be a whole number from 0 to 23.";
    }
  }

  SCENARIO_FIELDS.forEach((field) => {
    if (!form[field]) {
      nextErrors[field] = "Please choose a value.";
    }
  });

  return nextErrors;
}

function SelectField({ field, value, options, error, onChange }) {
  const meta = FIELD_META[field];
  return (
    <div className="field">
      <label htmlFor={field}>{meta.label}</label>
      <select id={field} value={value} onChange={(event) => onChange(field, event.target.value)}>
        {options.map((option) => (
          <option key={option} value={option}>{option}</option>
        ))}
      </select>
      {error ? <div className="validation-text">{error}</div> : null}
    </div>
  );
}

function NumberField({ value, error, onChange }) {
  const meta = FIELD_META.HOUR;
  return (
    <div className="field">
      <label htmlFor="HOUR">{meta.label}</label>
      <input
        id="HOUR"
        type="number"
        min="0"
        max="23"
        step="1"
        value={value}
        onChange={(event) => onChange("HOUR", event.target.value)}
      />
      {error ? <div className="validation-text">{error}</div> : null}
    </div>
  );
}

function ToggleField({ field, checked, onChange }) {
  const meta = FIELD_META[field];
  return (
    <label className={`toggle-card ${checked ? "active" : ""}`}>
      <input
        type="checkbox"
        checked={checked}
        onChange={(event) => onChange(field, event.target.checked)}
      />
      <div className="toggle-copy">
        <strong>{meta.label}</strong>
      </div>
    </label>
  );
}

function ModelSelector({ models, activeModel, isSwitching, onSwitch }) {
  return (
    <div className="model-grid">
      {Object.entries(models).map(([key, label], index) => (
        <button
          key={key}
          type="button"
          className={`model-card ${key === activeModel ? "active" : ""}`}
          disabled={isSwitching}
          onClick={() => onSwitch(key)}
        >
          <span className="model-role">Member Model {index + 1}</span>
          <span className="model-name">{label}</span>
        </button>
      ))}
    </div>
  );
}

function ResultCard({ result }) {
  const fatalPct = `${(result.prob_fatal * 100).toFixed(1)}%`;
  const nonFatalPct = `${(result.prob_non_fatal * 100).toFixed(1)}%`;
  const outcomeClass = result.prediction === 1 ? "fatal" : "non-fatal";

  return (
    <section className={`result-card ${outcomeClass}`}>
      <div className="result-header">
        <div>
          <div className="result-badge">{result.label}</div>
        </div>
        <div>
          <p className="result-score">Confidence: {result.confidence_pct}%</p>
          <div className="result-meta">
            Prediction generated using <strong>{result.model_used}</strong>. This card summarizes the model output for the chosen collision scenario.
          </div>
        </div>
      </div>

      <div className="progress-list">
        <div className="progress-row">
          <span>Fatal</span>
          <div className="progress-track">
            <div className="progress-fill fatal" style={{ width: fatalPct }}></div>
          </div>
          <strong>{fatalPct}</strong>
        </div>
        <div className="progress-row">
          <span>Non-Fatal</span>
          <div className="progress-track">
            <div className="progress-fill non-fatal" style={{ width: nonFatalPct }}></div>
          </div>
          <strong>{nonFatalPct}</strong>
        </div>
      </div>

      <div className="footer-note">
        Switch models to compare results using the same inputs.
      </div>
    </section>
  );
}

function App() {
  const [form, setForm] = useState(buildInitialFormState());
  const [errors, setErrors] = useState({});
  const [showRiskFactors, setShowRiskFactors] = useState(false);
  const [activeModel, setActiveModel] = useState(APP_CONFIG.activeModel || Object.keys(APP_CONFIG.modelRegistry || {})[0] || "");
  const [status, setStatus] = useState({
    kind: "ok",
    text: activeModel ? `Active model: ${APP_CONFIG.modelRegistry?.[activeModel] || activeModel}` : "Ready to test",
  });
  const [requestError, setRequestError] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isSwitching, setIsSwitching] = useState(false);
  const [result, setResult] = useState(null);

  function updateField(field, value) {
    setForm((current) => ({ ...current, [field]: value }));
    setErrors((current) => {
      if (!current[field]) return current;
      const nextErrors = { ...current };
      delete nextErrors[field];
      return nextErrors;
    });
  }

  async function handleModelSwitch(modelKey) {
    if (modelKey === activeModel) return;

    setIsSwitching(true);
    setRequestError("");
    setStatus({ kind: "", text: "Switching model..." });

    try {
      const response = await fetch(`${API_BASE}/set_model`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ model: modelKey }),
      });
      const data = await response.json();
      if (!response.ok || data.error) {
        throw new Error(data.error || "Unable to switch model.");
      }
      setActiveModel(modelKey);
      setResult(null);
      setStatus({ kind: "ok", text: `Active model: ${data.label}` });
    } catch (error) {
      setStatus({ kind: "error", text: "Model switch failed" });
      setRequestError(error.message);
    } finally {
      setIsSwitching(false);
    }
  }

  async function handlePredict() {
    const nextErrors = validateForm(form);
    setErrors(nextErrors);
    setRequestError("");

    if (Object.keys(nextErrors).length > 0) {
      setRequestError("Please fix the highlighted fields before predicting.");
      return;
    }

    setIsLoading(true);
    setStatus({ kind: "ok", text: "Running prediction..." });

    const payload = {
      ...form,
      HOUR: Number(form.HOUR),
    };

    try {
      const response = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json();
      if (!response.ok || data.error) {
        throw new Error(data.error || "Prediction request failed.");
      }
      setResult(data);
      setStatus({ kind: "ok", text: `Prediction ready from ${data.model_used}` });
    } catch (error) {
      setRequestError(error.message);
      setStatus({ kind: "error", text: "Prediction failed" });
    } finally {
      setIsLoading(false);
    }
  }

  function handleReset() {
    setForm(buildInitialFormState());
    setErrors({});
    setRequestError("");
    setResult(null);
    setStatus({
      kind: "ok",
      text: activeModel ? `Active model: ${APP_CONFIG.modelRegistry?.[activeModel] || activeModel}` : "Ready to test",
    });
  }

  return (
    <main className="app-shell">
      <section className="hero-grid">
        <div className="hero-card glass-card">
          <div className="eyebrow">COMP 247 Model Deployment</div>
          <h1>KSI Collision <span>Fatality Predictor</span></h1>
          <div className="hero-stats">
            <div className="hero-stat">
              <span className="hero-stat-label">Frontend</span>
              <span className="hero-stat-value">React + Flask</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-label">Outcome</span>
              <span className="hero-stat-value">Fatal / Non-Fatal</span>
            </div>
            <div className="hero-stat">
              <span className="hero-stat-label">Models</span>
              <span className="hero-stat-value">3 deployed models</span>
            </div>
          </div>
        </div>
      </section>

      <section className="workspace glass-card">
        <div className="workspace-head">
          <div>
            <h2>Prediction Workspace</h2>
          </div>
          <div className={`status-pill ${status.kind}`}>{status.text}</div>
        </div>

        <ModelSelector
          models={APP_CONFIG.modelRegistry || {}}
          activeModel={activeModel}
          isSwitching={isSwitching}
          onSwitch={handleModelSwitch}
        />

        <div className="content-single">
          <section className="panel-card">
            <h3>Step 1. Collision Details</h3>
            <div className="field-grid">
              {SCENARIO_FIELDS.map((field) => (
                <SelectField
                  key={field}
                  field={field}
                  value={form[field]}
                  options={APP_CONFIG.options?.[field] || []}
                  error={errors[field]}
                  onChange={updateField}
                />
              ))}
            </div>
          </section>

          <section className="panel-card compact-card">
            <h3>Step 2. Time</h3>
            <div className="field-grid" style={{ gridTemplateColumns: "minmax(0, 240px)" }}>
              <NumberField value={form.HOUR} error={errors.HOUR} onChange={updateField} />
            </div>
          </section>

          <section className="panel-card compact-card">
            <button
              type="button"
              className={`collapse-toggle ${showRiskFactors ? "open" : ""}`}
              onClick={() => setShowRiskFactors((current) => !current)}
            >
              <span>Step 3. Optional Risk Factors</span>
              <span>{showRiskFactors ? "Hide" : "Show"}</span>
            </button>
            {showRiskFactors ? (
              <div className="toggle-grid">
                {RISK_FLAGS.map((field) => (
                  <ToggleField
                    key={field}
                    field={field}
                    checked={Boolean(form[field])}
                    onChange={updateField}
                  />
                ))}
              </div>
            ) : null}
          </section>
        </div>

        {requestError ? <div className="error-banner">{requestError}</div> : null}

        <div className="action-row">
          <button className="button button-primary" type="button" disabled={isLoading || isSwitching} onClick={handlePredict}>
            {isLoading ? "Predicting..." : "Predict Fatality Risk"}
          </button>
          <button className="button button-secondary" type="button" disabled={isLoading || isSwitching} onClick={handleReset}>
            Reset Form
          </button>
        </div>

        {result ? <ResultCard result={result} /> : null}
      </section>
    </main>
  );
}

ReactDOM.createRoot(document.getElementById("root")).render(<App />);
