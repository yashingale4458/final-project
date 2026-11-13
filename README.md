# 1f151e50-6931-4d90-8b8e-ae724fa171c3-cefef302-df98-4a92-ac58-03c011301e9e
https://sonar.server.examly.io/dashboard?id=iamneo-production-2_1f151e50-6931-4d90-8b8e-ae724fa171c3-cefef302-df98-4a92-ac58-03c011301e9e&codeScope=overall

---

<div align="center">

# ⚡ PhaseWise AI
### Home Low-Carbon Connection Review & Decision Platform

[![Tests](https://img.shields.io/badge/pytest-50%2F50%20passing%20(100%25)-success?style=for-the-badge&logo=pytest)](docs/DEVELOPMENT_AND_OPERATIONS.md)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-009688?style=for-the-badge&logo=fastapi)](docs/API_REFERENCE.md)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=for-the-badge&logo=streamlit)](docs/ARCHITECTURE.md)
[![LangGraph](https://img.shields.io/badge/LangGraph-Multi--Agent-1C75BC?style=for-the-badge)](docs/ARCHITECTURE.md)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-Flash--Lite-4285F4?style=for-the-badge&logo=google)](docs/DEVELOPMENT_AND_OPERATIONS.md)
[![License](https://img.shields.io/badge/license-Proprietary-blue?style=for-the-badge)](#)

*An enterprise AI review platform built for **Merrowdale Power Networks** to streamline, validate, and accelerate residential low-carbon technology connections while safeguarding the distribution electricity grid.*

</div>

---

## 🌟 Overview

As the UK transitions towards Net Zero, thousands of households are installing low-carbon technologies (LCTs): **Electric Vehicle (EV) Charge Points**, **Heat Pumps**, **Vehicle-to-Grid (V2G) systems**, **Solar PV**, and **Battery Storage (BESS)**. Every installation must be reviewed by the local Distribution Network Operator (DNO) to prevent service cut-out fuse failures, substation transformer overloads, voltage imbalances, and reverse power flow hazards.

**PhaseWise AI** is an end-to-end multi-agent AI system that unifies application intake, normalizes dirty multi-portal submissions, performs deterministic engineering screening, predicts grid reinforcement needs and job durations using ML, indexes engineering standards in a vector database, orchestrates LangGraph multi-agent reviews, and gates decisions behind human engineers.

---

## 🚀 Key Features

- **Multi-Portal Ingestion & Cleaning**: Ingests disparate submission streams from **Wattcroft**, **InstallerLine**, and **SiteWren** portals, normalizing references, electrical units (A, kW, W, kVA), power factors, and phase configurations.
- **Deterministic Screening Engine**: Validates equipment ratings, service cut-out headroom, earthing configurations (PME / TN-C-S), open-PEN detection, and raises network triggers for high-risk feeds.
- **Machine Learning Suite**:
  - *Network Work Model*: Held-out ROC AUC **0.8540** (vs. 0.82 threshold).
  - *Duration Model*: Held-out $R^2$ **0.7851**, MAE **3.55 working days** (vs. 0.70 threshold).
  - *Sentence Classifier*: Held-out Macro $F_1$ **0.9757** (vs. 0.95 threshold).
- **RAG Knowledge Base**: Embedded ChromaDB indexing 16 official Merrowdale engineering standards and guides using sentence-transformers.
- **LangGraph Multi-Agent Review Workflow**:
  - `review_planner_agent`: Formulates technical review plan based on coordinator requests.
  - `requirements_analyst_agent`: Validates requirements against governing standard documents.
  - `evidence_analyst_agent`: Parses installer statements and flags rating contradictions.
  - `risk_analyst_agent`: Synthesises ML network probabilities and schedule risks.
  - `policy_synthesiser_agent`: Drafts formal installer notices, household letters, and briefs.
- **Human-in-the-Loop Decision Gate**: Review states are check-pointed via `SqliteSaver` in `database/checkpoint.db`. Reviews pause at the gate until an independent Connections Engineer approves, returns, or refers the application.
- **Role-Based Access Control (RBAC)**: Secure access tailored for Connections Engineers, Connections Coordinators, and Customer Service Advisors.
- **Modern Operations Dashboard**: High-performance Streamlit UI featuring Lucide SVG icons, custom typography, responsive metric cards, and live audit telemetry.

---

## 🏛️ System Architecture

```
                                    +-----------------------------------------+
                                    |        STREAMLIT DASHBOARD (8501)       |
                                    |  Queue | Detail | Gate | Audit | Chat   |
                                    +--------------------+--------------------+
                                                         | REST (X-Session-Token)
                                                         v
                                    +-----------------------------------------+
                                    |          FASTAPI BACKEND (8081)         |
                                    |  Auth Guard | Routing | Error Handlers  |
                                    +----+----------+----------+----------+---+
                                         |          |          |          |
                      +------------------+          |          |          +------------------+
                      v                             v          v                             v
           +--------------------+         +------------+    +------------+         +--------------------+
           |   DETERMINISTIC    |         |  CHROMA DB |    |  SCIKIT-   |         |     LANGGRAPH      |
           |     SCREENING      |         | VECTOR RAG |    |  LEARN ML  |         |    MULTI-AGENT     |
           | Cut-out & Standards|         |  16 Docs   |    |  3 Models  |         | 5 Agents + Gate    |
           +--------------------+         +------------+    +------------+         +--------------------+
                      |                             |          |                             |
                      +-----------------------------+----+-----+-----------------------------+
                                                         |
                                                         v
                                    +-----------------------------------------+
                                    |            PERSISTENCE LAYER            |
                                    | phasewise.db (SQLite) | checkpoint.db   |
                                    +-----------------------------------------+
```

---

## 📁 Repository Structure

```text
PhasewiseAI/
├── .env.example                     # Environment template (Gemini API credentials)
├── README.md                        # Master project documentation
├── sonar-project.properties         # SonarQube scanner properties
├── docs/                            # Comprehensive Technical Documentation
│   ├── ARCHITECTURE.md              # System design, multi-agent graph, & data flow
│   ├── API_REFERENCE.md             # Complete REST API manual with request/response schemas
│   ├── DATABASE_SCHEMA.md           # ER diagram, table schemas, & data dictionary
│   └── DEVELOPMENT_AND_OPERATIONS.md# Installation, training, local run, & prod deployment
└── Project/                         # Core Application Source Code
    ├── access/                      # RBAC, password hashing, and permission guards
    ├── agentic/                     # LangGraph workflow, nodes, state, & agent prompts
    ├── api/                         # FastAPI routes, Pydantic schemas, & exception handlers
    ├── chat/                        # RAG connection assistant & conversation memory
    ├── config/                      # System constants, standards definitions, & styling
    ├── dashboard/                   # Streamlit pages, UI design system, & API client
    ├── data/                        # Raw datasets, live queues, and 16 engineering PDFs
    ├── data_pipeline/               # Ingestion, unit normalization, & data quality profiling
    ├── database/                    # SQLite relational DB, vector store, & checkpoints
    ├── knowledge/                   # PDF extraction, chunk splitting, & ChromaDB indexing
    ├── modeling/                    # Scikit-learn pipelines, cross-validation, & registry
    ├── persistence/                 # SQLAlchemy 2.0 ORM models & repository operations
    ├── tools/                       # Screening, decision, evidence, notice, & prediction tools
    ├── tests/                       # Unit & component test suites
    ├── tests.py                     # Master 50-test evaluation suite
    ├── run_database_setup.py        # Entry point 1: Database & seed accounts
    ├── run_data_pipeline.py         # Entry point 2: Clean extracts & quality profiling
    ├── run_model_training.py        # Entry point 3: Train & validate ML models
    ├── run_knowledge_indexing.py    # Entry point 4: Index 16 PDF standards
    ├── run_api_service.py           # Entry point 5: Start FastAPI service (port 8081)
    └── run_dashboard.py             # Entry point 6: Start Streamlit dashboard (port 8501)
```

---

## ⚡ Quick Start

### 1. Prerequisites
- Python 3.10 to 3.14
- Google Gemini API Key

### 2. Configure Environment
```bash
cd Project
cp ../.env.example .env
# Edit .env and insert your GEMINI_API_KEY
```

### 3. Run Pipeline Setup
Run the four preparation scripts in sequence:
```bash
python run_database_setup.py      # Creates database & 14 seed accounts
python run_data_pipeline.py       # Cleans extracts & generates quality report
python run_model_training.py      # Trains & verifies the 3 ML models
python run_knowledge_indexing.py  # Builds ChromaDB vector store
```

### 4. Start Services
In Terminal 1 (FastAPI Backend):
```bash
python run_api_service.py
# API running at http://localhost:8081
# Swagger UI at http://localhost:8081/docs
```

In Terminal 2 (Streamlit Dashboard):
```bash
python -m streamlit run run_dashboard.py
# Dashboard running at http://localhost:8501
```

### 5. Run with Docker

The image prepares the SQLite schema, cleaned datasets, ML models, and Chroma index during
the build. Keep `PHASEWISE_OFFLINE_LLM` unset (or `0`) for real Gemini calls; use `1` only
for a deterministic local demo.

```bash
docker build -t phasewise-ai .
docker run --rm -p 8501:8501 \
  -e GEMINI_API_KEY="$GEMINI_API_KEY" \
  -e GEMINI_MODEL="gemini-2.5-flash-lite" \
  phasewise-ai
```

Open `http://localhost:8501`. The API stays internal to the container on port `8081` and
is exposed to the dashboard through the internal service URL. The sidebar shows whether the
current process is using live Gemini or the explicit offline demo mode.

---

## 🧪 Testing & Validation

Execute the master evaluation suite (50 comprehensive tests):
```bash
cd Project
PYTHONPATH=. pytest tests.py
```
*Result:* **50 passed out of 50 (100%)**

Execute component unit tests:
```bash
cd Project
PYTHONPATH=. pytest tests/
```
*Result:* **17 passed, 1 skipped (0 failures)**

---

## 👥 Seed Staff Accounts

| Username | Role | Password | Permitted Areas |
|---|---|---|---|
| `amira` | Connections Engineer | *(Seeded)* | Full Access (Queue, Detail, Decision Gate, Audit, Chat) |
| `lewis` | Connections Engineer | *(Seeded)* | Full Access (Queue, Detail, Decision Gate, Audit, Chat) |
| `ravi` | Connections Coordinator | *(Seeded)* | Queue, Detail, Start Reviews, Audit, Chat |
| `hannah` | Connections Coordinator | *(Seeded)* | Queue, Detail, Start Reviews, Audit, Chat |
| `zara` | Customer Service Advisor | *(Seeded)* | Connection Chat only |
| `joel` | Customer Service Advisor | *(Seeded)* | Connection Chat only |

---

## 📖 Detailed Technical Documentation

- **[System Architecture & Multi-Agent Design](docs/ARCHITECTURE.md)**: Deep dive into the LangGraph state machine, data cleaning pipeline, RAG embeddings, and security architecture.
- **[REST API Reference](docs/API_REFERENCE.md)**: Full endpoint specifications, Pydantic schemas, request/response models, and error statuses.
- **[Database Schema & Dictionary](docs/DATABASE_SCHEMA.md)**: SQLAlchemy models, SQLite ER diagram, column types, and checkpoint tables.
- **[Development & Operations Guide](docs/DEVELOPMENT_AND_OPERATIONS.md)**: Step-by-step setup, test procedures, Nginx reverse proxy configuration, systemd services, and production deployment considerations.
