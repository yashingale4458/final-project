# PhaseWise AI — Development, Operations & Deployment Guide

This guide describes how to configure, train, test, operate, and deploy the **PhaseWise AI** application suite.

---

## 1. System Requirements & Prerequisites

### 1.1 Hardware Specifications
- **CPU**: 4 cores minimum (8 cores recommended for data cleaning and model training).
- **RAM**: 8 GB minimum (16 GB recommended for sentence embeddings & scikit-learn training).
- **Disk**: 5 GB free storage for database, embeddings index, and trained models.

### 1.2 Software Requirements
- **Operating System**: Linux (Ubuntu 20.04+ / Debian 11+), macOS 12+, or Windows 11 with WSL2.
- **Python**: Python 3.10 to 3.14.
- **Package Manager**: `pip` (or `poetry` / `uv`).

---

## 2. Environment Configuration

PhaseWise AI relies on Google Gemini models for the LangGraph multi-agent workflow and conversational assistant.

1. Copy the example environment file:
   ```bash
   cp .env.example .env
   ```

2. Populate `.env` with your API credentials:
   ```ini
   # Required: Google Gemini API Key
   GEMINI_API_KEY="AIzaSy..."

   # Model name (Production recommendation: gemini-3.5-flash-lite or gemini-3.8-flash)
   GEMINI_MODEL="gemini-3.5-flash-lite"
   ```

---

## 3. Installation & Setup Pipeline

The system requires running four preparation steps in order before launching services:

```
+--------------------------+
| 1. run_database_setup.py |  --> Creates tables & 14 seed staff accounts
+--------------------------+
             |
             v
+--------------------------+
| 2. run_data_pipeline.py  |  --> Ingests, normalizes, & cleans 5 raw extracts
+--------------------------+
             |
             v
+--------------------------+
| 3. run_model_training.py |  --> Trains Network Work, Duration, & Sentence models
+--------------------------+
             |
             v
+------------------------------+
| 4. run_knowledge_indexing.py |  --> Builds ChromaDB vector store over 16 PDFs
+------------------------------+
             |
             v
+--------------------------+
| 5. run_api_service.py    |  --> Starts FastAPI backend (port 8081)
+--------------------------+
             |
             v
+--------------------------+
| 6. run_dashboard.py      |  --> Starts Streamlit frontend (port 8501)
+--------------------------+
```

### Step 1: Database Setup
Initializes `database/phasewise.db`, builds the 6 relational tables, and provisions the 14 seed user accounts with hashed credentials.
```bash
python run_database_setup.py
```
*Expected Output:*
```text
database: database/phasewise.db
tables: agent_runs, application_reviews, chat_turns, llm_calls, review_decisions, user_accounts
accounts created: 14, already present: 0
```

### Step 2: Data Pipeline Cleaning
Cleans raw extracts from Wattcroft, InstallerLine, and SiteWren portals, performs deduplication, harmonizes power/unit formats, and generates quality profiling reports in `output/data_quality_report.json`.
```bash
python run_data_pipeline.py
```
*Expected Output:*
```text
application_history_train       31750 received ->  29597 kept  (2153 removed)
application_history_eval         7933 received ->   7407 kept  (526 removed)
connection_logs_train           30205 received ->  29570 kept  (635 removed)
connection_logs_eval             7563 received ->   7389 kept  (174 removed)
statement_sentences             36652 received ->  35725 kept  (927 removed)
quality report written to output/data_quality_report.json
```

### Step 3: Model Training
Trains the three machine learning models. Models are verified against strict minimum held-out evaluation thresholds before saving to `output/models/`.
```bash
python run_model_training.py
```
*Quality Gates Enforced:*
- **Network Work Model**: Held-out ROC AUC $\ge 0.82$ (Achieved $\approx 0.854$).
- **Duration Model**: Held-out $R^2 \ge 0.70$ (Achieved $\approx 0.785$, MAE $\approx 3.55$ days).
- **Sentence Model**: Held-out Macro $F_1 \ge 0.95$ (Achieved $\approx 0.976$).

### Step 4: Knowledge Base Indexing
Extracts text from 16 connection standards and guides in `data/docs/`, splits them into semantic chunks, and builds the ChromaDB index in `database/chroma_store/`.
```bash
python run_knowledge_indexing.py
```

---

## 4. Running the Application Locally

### 4.1 Launch the FastAPI Backend
Start the REST service on port 8081:
```bash
python run_api_service.py
```
The service verifies all required databases, models, schemas, and vector stores before starting. Access the interactive API documentation at:
```
http://localhost:8081/docs
```

### 4.2 Launch the Streamlit Dashboard
In a separate terminal, launch the dashboard:
```bash
python -m streamlit run run_dashboard.py
```
The dashboard will open automatically in your browser at:
```
http://localhost:8501
```

### 4.3 Default Staff Sign-In Credentials
You can sign in using any of the seeded staff accounts:

| Username | Role | Primary Permissions |
|---|---|---|
| `amira` | Connections Engineer | Queue, Detail, Decision Gate, Audit Trail, Chat |
| `lewis` | Connections Engineer | Queue, Detail, Decision Gate, Audit Trail, Chat |
| `ravi` | Connections Coordinator | Queue, Detail, Start Reviews, Audit Trail, Chat |
| `hannah` | Connections Coordinator | Queue, Detail, Start Reviews, Audit Trail, Chat |
| `zara` | Customer Service Advisor | Connection Chat only |
| `joel` | Customer Service Advisor | Connection Chat only |

*(All seed accounts use standard test credentials provisioned during setup)*.

---

## 5. Testing & Validation

### 5.1 Running the Official Evaluation Suite
The project includes a 50-test comprehensive benchmark suite in `tests.py`:
```bash
PYTHONPATH=. pytest tests.py
```
*Expected Outcome:* **50 passed out of 50 (100%)**.

### 5.2 Running the Unit & Component Tests
Run individual module test suites:
```bash
PYTHONPATH=. pytest tests/
```
*Expected Outcome:* **17 passed, 1 skipped (0 failures)**.

---

## 6. Production Deployment Considerations

### 6.1 Reverse Proxy & SSL (Nginx)
In production, place FastAPI and Streamlit behind an Nginx reverse proxy with TLS termination:
```nginx
# FastAPI Backend
server {
    server_name api.phasewise.internal;
    listen 443 ssl http2;

    ssl_certificate /etc/ssl/certs/phasewise.crt;
    ssl_certificate_key /etc/ssl/private/phasewise.key;

    location / {
        proxy_pass http://127.0.0.1:8081;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}

# Streamlit Dashboard
server {
    server_name phasewise.internal;
    listen 443 ssl http2;

    ssl_certificate /etc/ssl/certs/phasewise.crt;
    ssl_certificate_key /etc/ssl/private/phasewise.key;

    location / {
        proxy_pass http://127.0.0.1:8501;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_read_timeout 86400;
    }
}
```

### 6.2 Systemd Service Configuration

**FastAPI Service (`/etc/systemd/system/phasewise-api.service`):**
```ini
[Unit]
Description=PhaseWise AI FastAPI Service
After=network.target

[Service]
Type=simple
User=naruto
WorkingDirectory=/home/naruto/Desktop/PhasewiseAI/Project
EnvironmentFile=/home/naruto/Desktop/PhasewiseAI/Project/.env
ExecStart=/usr/bin/python3 run_api_service.py
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

**Streamlit Dashboard Service (`/etc/systemd/system/phasewise-dashboard.service`):**
```ini
[Unit]
Description=PhaseWise AI Streamlit Dashboard
After=phasewise-api.service

[Service]
Type=simple
User=naruto
WorkingDirectory=/home/naruto/Desktop/PhasewiseAI/Project
EnvironmentFile=/home/naruto/Desktop/PhasewiseAI/Project/.env
ExecStart=/usr/bin/python3 -m streamlit run run_dashboard.py --server.port 8501 --server.address 0.0.0.0
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

---

## 7. Security & Governance

1. **Authentication & Session Security**:
   - Passwords hashed using unique 16-byte random salts.
   - Session tokens generated using `secrets.token_urlsafe(32)`.
   - Single-session enforcement: Signing in on a new device immediately invalidates previous tokens.
2. **Conflict of Interest Protection**:
   - Connections Engineers cannot decide reviews that they opened as a coordinator.
3. **Information Barrier for Customer Service**:
   - Customer Service Advisors are restricted from viewing unfinalized reviews, predictive network risk scores, or duration delays.
4. **Token Budget & Telemetry**:
   - Every Gemini call logs prompt tokens, completion tokens, latency, and agent ID to `llm_calls` table for audit compliance.
