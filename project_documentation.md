# AI Predictive Maintenance Platform (Operations Center)
## Complete Project Documentation

This documentation serves as a comprehensive study and reference guide for the **AI Predictive Maintenance Platform** built on the NASA CMAPSS Turbofan Engine dataset.

---

## 1. Problem Statement

In modern manufacturing, aviation, and heavy industries, machinery breakdowns carry massive operational and financial penalties:
1. **Unscheduled Downtime**: Sudden equipment failures halt production lines, leading to OEE (Overall Equipment Effectiveness) drops and missed SLA targets.
2. **Compounded Damage**: Operating a degrading component (like a failing bearing or turbine blade) often causes catastrophic secondary damage to adjacent parts, multiplying repair costs.
3. **Ineffective Maintenance Routines**:
   - *Reactive Maintenance (Run-to-Failure)*: Waiting for breakdowns is expensive and dangerous.
   - *Preventive Maintenance (Time-based/Schedule-based)*: Servicing machines at arbitrary intervals (e.g., every 3 months) leads to replacing parts that still have remaining useful life, or conversely, missing a failure that happens before the schedule.

### The Solution: Predictive Maintenance (PdM)
By ingesting continuous telemetry streams from industrial sensors (temperature, pressure, vibration, rotational speed) and applying machine learning models, we can:
- Forecast the **Remaining Useful Life (RUL)** of assets.
- Classify whether a machine is operating in a **normal state**, **warning state** (degradation started), or **critical state** (failure imminent within the next 30 cycles).
- Schedule targeted maintenance only when required, preventing breakdowns while maximizing part longevity.

---

## 2. Dataset Context (NASA CMAPSS)

The platform is built around the **NASA Turbofan Engine Degradation Simulation Dataset (CMAPSS)**:
- **Files**: `train_FD001.csv` containing 20,633 logs for 100 engines.
- **Data Structure**: Each row represents an operational cycle for a specific engine, containing:
  - `unit_number`: Engine identifier (e.g., Engine 1, Engine 2).
  - `time_in_cycles`: Sequential cycle steps.
  - `op_setting_1, op_setting_2, op_setting_3`: Three variables controlling operational settings.
  - `sensor_1` to `sensor_21`: 21 physical sensor readings monitoring variables such as fan speed, bypass ratio, core temperatures, fuel flow, and pressure ratios.
- **Failure Condition**: Each engine starts in a healthy state and degrades gradually over time until it fails. The final cycle recorded in the training set for any engine represents the exact cycle of mechanical failure.

---

## 3. Technology Stack

The platform is constructed using a modern, production-grade decoupled architecture:

| Tier | Technology | Role |
| :--- | :--- | :--- |
| **Frontend** | React 18 (Vite), Vanilla CSS | FUTURISTIC dark industrial console dashboard, live SVG line charts, simulation controller. |
| **Backend** | FastAPI, Uvicorn | High-performance, async REST endpoints for telemetry ingestion, inference, alert triggers, and retraining. |
| **Database** | PostgreSQL / SQLite | SQLAlchemy ORM. Logs telemetry, ML predictions, and alert notifications. Seeding tables from NASA CSV. |
| **Machine Learning** | Scikit-learn, XGBoost, Pandas, Numpy | Preprocessing, rolling window calculations, training classifiers, and model serializations. |
| **Monitoring** | MLflow | local SQLite backend, tracking training runs, F1-scores, accuracy, and registering the active model. |
| **Orchestration**| Docker, Docker Compose, Nginx | Containerizing backend (FastAPI), database (PostgreSQL), and serving React static builds via Nginx. |

---

## 4. Machine Learning & Feature Engineering

### Preprocessing & Target Formulation
1. **RUL Extraction**: For each cycle $t$ of unit $i$, the Remaining Useful Life is calculated as:
   $$\text{RUL}_{i,t} = \max(\text{cycle}_{i}) - \text{cycle}_{i,t}$$
2. **Labeling**: To build a binary classifier that predicts failure within the next 30 cycles (the standard threshold for scheduling engine maintenance):
   $$\text{failure}_{i,t} = \begin{cases} 
      1, & \text{if } \text{RUL}_{i,t} \le 30 \\ 
      0, & \text{otherwise} 
   \end{cases}$$
3. **Data Cleaning**: Discarded 7 constant/invariant sensors (1, 5, 6, 10, 16, 18, and 19) to eliminate noise and reduce dimensional complexity.

### Feature Engineering
To capture degradation *trends* rather than just instantaneous values, we calculate rolling features over a sliding window of the **last 5 cycles** for the 12 critical sensors:
- **Rolling Mean**: $\mu = \frac{1}{k}\sum_{j=0}^{k-1} x_{t-j}$
- **Rolling Standard Deviation**: $\sigma = \sqrt{\frac{1}{k-1}\sum_{j=0}^{k-1} (x_{t-j} - \mu)^2}$
These rolling parameters allow the model to recognize acceleration in vibration or temperature drift curves.

### Model Benchmarks
We trained two classifiers using the engineered features and split the units (80% train / 20% validation) to prevent data leakage:

1. **Random Forest Classifier**:
   - **Validation Accuracy**: **96.07%**
   - **Precision**: **89.52%**
   - **Recall**: **84.03%**
   - **F1-Score**: **86.69%** (Primary metric selected for balance between precision and recall)
   - **ROC-AUC**: **0.9901**
2. **XGBoost Classifier**:
   - **Validation Accuracy**: **95.33%**
   - **Precision**: **86.82%**
   - **Recall**: **81.77%**
   - **F1-Score**: **84.22%**
   - **ROC-AUC**: **0.9868**

*Model Selection*: The **Random Forest** model was chosen and registered as the primary inference engine due to its superior F1 performance.

---

## 5. System Deliverables & Outputs

When the application is running, the following systems work in synchronization:

### 1. Ingestion & Prediction API
The FastAPI server exposes REST endpoints to write sensor logs, query database history, score failure risk, and resolve alarms.
* `POST /api/telemetry`: Ingests sensor data. If model binaries are not trained yet, it falls back to a rule-based physics heuristic model so the server never crashes.
* `GET /api/dashboard/stats`: Aggregates database entries to compute system metrics.
* `GET /api/simulation/history/{unit_number}`: Supplies chronological arrays for chart lines.

### 2. Operational Seeding & Simulator
* **Automatic Database Seeding**: On the first start, the database parses `train_FD001.csv` and imports historical cycles for Engines 1, 2, and 3 (up to cycles 110, 100, and 90) so that the UI is populated out-of-the-box.
* **Simulator Controller** (`POST /api/simulation/step`): Next to the fleet asset list in the UI, clicking the "Step" button advances the selected machine's lifespan by exactly 1 cycle. The backend loads the next real telemetry row from the NASA CSV, evaluates it via the ML model, updates the charts, and triggers alerts.

### 3. Automated Alert System
When a telemetry row is processed and the predicted breakdown probability rises:
- **Risk $\ge$ 35%**: Creates a `WARNING` database entry, logs warning messages, and broadcasts alerts.
- **Risk $\ge$ 70%**: Creates a `CRITICAL` database entry, prints a simulated emergency warning email text to the terminal, and logs a mock Slack channel payload.
- **Resolution Filter**: Once a user clicks "Clear All System Alerts" or "Resolve", the status is updated to `resolved = True` in the database, and the alerts are filtered out of the active dashboard view.

### 4. Database Schema & SQL Views
The Postgres schema (`database/init.sql`) defines indices to accelerate queries on `unit_number` and timestamps, and builds a SQL analytics view:
```sql
CREATE OR REPLACE VIEW view_machine_health_summary AS
SELECT 
    t.unit_number,
    MAX(t.time_in_cycles) as total_cycles,
    AVG(p.failure_probability) as avg_failure_probability,
    (
        SELECT p2.failure_probability 
        FROM prediction_logs p2 
        WHERE p2.unit_number = t.unit_number 
        ORDER BY p2.timestamp DESC LIMIT 1
    ) as current_failure_probability,
    COUNT(CASE WHEN a.resolved = FALSE THEN 1 END) as open_alerts_count
FROM telemetry_logs t
LEFT JOIN prediction_logs p ON t.id = p.telemetry_id
LEFT JOIN alerts a ON t.unit_number = a.unit_number
GROUP BY t.unit_number;
```

---

## 6. How to Run the Project

### Local Installation (No Containers)
1. Install Python packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Train the model (outputs `models/model.joblib` and tracks parameters in MLflow):
   ```bash
   python models/train.py
   ```
3. Run the FastAPI backend:
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
4. Run the React frontend:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Open `http://localhost:3000/` (or `http://localhost:5173/`).

### Container Launch (Orchestrated Stack)
From the `deployment/` directory:
```bash
docker-compose up --build
```
- **React Frontend**: `http://localhost:3000`
- **FastAPI Endpoints**: `http://localhost:8000/docs`
- **MLflow Tracking Server**: `http://localhost:5000`
- **PostgreSQL**: Local port `5432`
