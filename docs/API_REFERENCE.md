# PhaseWise AI — REST API Reference Manual

The PhaseWise AI REST service is built with **FastAPI** and runs on **port 8081** by default.
All endpoints require authentication via the `X-Session-Token` header, except `/auth/signin` and `/auth/signout`.

Interactive Swagger UI documentation is accessible at:
```
http://localhost:8081/docs
```

---

## 1. Authentication & Session Management

### 1.1 Sign In
Authenticates staff members and returns a cryptographically secure session token, along with permitted UI pages and functional areas.

- **Endpoint**: `POST /auth/signin`
- **Access**: Public
- **Request Headers**: `Content-Type: application/json`

#### Request Body
```json
{
  "username": "amira",
  "password": "password123"
}
```

#### Response (200 OK)
```json
{
  "username": "amira",
  "display_name": "Amira Okafor",
  "role": "Connections Engineer",
  "session_token": "IcyoceTNe3M5_aFVLug7gGDK3htX8xk6",
  "pages": [
    "Application queue",
    "Application detail",
    "Decision queue",
    "Audit trail",
    "Connection chat"
  ],
  "areas": [
    "queue",
    "detail",
    "reviews",
    "decisions",
    "audit",
    "chat"
  ]
}
```

#### Error Responses
- `401 Unauthorized`: When username or password is incorrect.
  ```json
  {
    "detail": "Username and password did not match a known account"
  }
  ```

---

### 1.2 Sign Out
Invalidates an active session token. A terminated token cannot be reused for any subsequent request.

- **Endpoint**: `POST /auth/signout`
- **Access**: Authenticated
- **Request Headers**: `X-Session-Token: <token>`

#### Response (200 OK)
```json
{
  "username": "amira",
  "signed_out": true
}
```

#### Error Responses
- `401 Unauthorized`: When the session token is missing, expired, or invalid.

---

## 2. Queue & Application Inspection

### 2.1 List Application Queue
Returns the ranked list of open applications waiting in the central queue. Ranked according to Merrowdale priority policy: **Heating-failed heat pumps first**, followed by all other technologies ordered by application reference.

- **Endpoint**: `GET /queue`
- **Access**: Connections Coordinator, Connections Engineer
- **Request Headers**: `X-Session-Token: <token>`
- **Query Parameters**:
  - `technology` (required, string): Filter by technology. Permitted values:
    - `"All"`: All technologies
    - `"EVCP"`: Electric Vehicle Charge Points
    - `"HP"`: Heat Pumps
    - `"V2G"`: Vehicle-to-Grid
    - `"PV"`: Solar PV
    - `"BESS"`: Battery Storage

#### Response (200 OK)
```json
[
  {
    "queue_position": 1,
    "application_reference": "LCT0052125",
    "technology": "HP",
    "technology_label": "Heat pump",
    "source_portal": "Wattcroft",
    "premises_type": "Domestic",
    "urgency": "Emergency - Heating System Failed",
    "document_count": 2,
    "requirements_met_count": 4,
    "requirements_not_met_count": 0,
    "requirements_not_documented_count": 0,
    "network_trigger_count": 0,
    "rating_stated": true,
    "assessment": "Approve",
    "decisions_open": ["Approve"],
    "network_work_likelihood": 0.1245,
    "likelihood_band": "Low",
    "expected_working_days": 14.5,
    "delay_expected": false,
    "disagreement_count": 0
  }
]
```

#### Error Responses
- `401 Unauthorized`: Missing or invalid session token.
- `403 Forbidden`: User role is not permitted (e.g., Customer Service Advisor).
- `422 Unprocessable Entity`: Missing `technology` parameter or unknown technology code.

---

### 2.2 Get Application Detail
Fetches the complete engineering context of a single application prior to spending any language model calls. Includes full form responses, deterministic screening outcomes, evidence extracted from attached statements, ML predictions, and duration expectations.

- **Endpoint**: `GET /queue/{application_reference}`
- **Access**: Connections Coordinator, Connections Engineer
- **Request Headers**: `X-Session-Token: <token>`
- **Path Parameters**:
  - `application_reference` (string, e.g. `LCT0052125`)

#### Response (200 OK)
```json
{
  "queue_position": 1,
  "application": {
    "application_reference": "LCT0052125",
    "technology": "HP",
    "source_portal": "Wattcroft",
    "premises_type": "Domestic",
    "supply_cut_out_rating_amps": "80",
    "heat_pump_rating_amps": "24",
    "power_factor": "0.95",
    "earthing_type": "TN-C-S"
  },
  "application_documents": [
    {
      "document_id": "DOC-7721",
      "document_title": "Heat Pump Commissioning Certificate",
      "sentence_count": 6
    }
  ],
  "screening": {
    "technology": "HP",
    "standard_document": "Connection-Standard-Heat-Pump.pdf",
    "assessment": "Approve",
    "rating_stated": true,
    "decisions_open": ["Approve"],
    "network_triggers": [],
    "requirements": [
      {
        "requirement_name": "Maximum demand under cutout",
        "requirement_sentence": "The heat pump demand combined with existing load must not exceed the cut-out fuse rating.",
        "result": "Met",
        "blank_answers": []
      }
    ]
  },
  "evidence_findings": [
    {
      "document_id": "DOC-7721",
      "sentence": "The heat pump model is rated at 24A full-load running current.",
      "sentence_type": "RATING",
      "requirements_borne_on": ["Rating stated"]
    }
  ],
  "disagreements": [],
  "likelihood": {
    "network_work_likelihood": 0.1245,
    "likelihood_band": "Low"
  },
  "duration": {
    "expected_working_days": 14.5,
    "delay_expected": false
  },
  "duration_text": "14.5 working days (on time)"
}
```

#### Error Responses
- `404 Not Found`: Unknown application reference.

---

## 3. Multi-Agent Review Workflow

### 3.1 Start Review
Initiates the 5-agent review workflow for an open application. Executes the Review Planner, Requirements Analyst, Evidence Analyst, and Risk Analyst agents, then **pauses at the human decision gate**.

- **Endpoint**: `POST /reviews`
- **Access**: Connections Coordinator, Connections Engineer
- **Request Headers**: `X-Session-Token: <token>`, `Content-Type: application/json`

#### Request Body
```json
{
  "application_reference": "LCT0052125",
  "coordinator_request": "Verify heat pump full-load current and supply cut-out headroom."
}
```

#### Response (201 Created)
```json
{
  "review_id": "REV-LCT0052125-9A4F",
  "run_id": "RUN-20261112-9A4F",
  "application_reference": "LCT0052125",
  "technology": "HP",
  "coordinator_id": "ravi",
  "status": "awaiting_decision",
  "assessment": "Approve",
  "decisions_open": ["Approve"],
  "recommended_decision": "Approve",
  "recommendation_text": "Based on the technical screening and statement review, the heat pump meets all Connection Standard requirements...",
  "decision": null,
  "decided_by": null,
  "decided_by_role": null,
  "decision_reason": null,
  "llm_call_count": 4
}
```

#### Error Responses
- `404 Not Found`: Application reference not found in active queue.
- `409 Conflict`: Application has already been reviewed.
- `502 Bad Gateway`: External LLM API unavailable or malformed response.

---

### 3.2 List Reviews
Retrieves reviews matching specified statuses.

- **Endpoint**: `GET /reviews`
- **Access**: Connections Coordinator, Connections Engineer
- **Request Headers**: `X-Session-Token: <token>`
- **Query Parameters**:
  - `status_value` (list of strings): Multiple status values permitted:
    - `"awaiting_decision"`
    - `"approved"`
    - `"returned"`
    - `"referred"`

#### Response (200 OK)
```json
[
  {
    "review_id": "REV-LCT0052125-9A4F",
    "application_reference": "LCT0052125",
    "technology": "HP",
    "coordinator_id": "ravi",
    "status": "awaiting_decision",
    "recommended_decision": "Approve",
    "llm_call_count": 4
  }
]
```

---

### 3.3 Submit Review Decision (Human-in-the-Loop)
Records the final connection decision by an engineer and resumes the paused LangGraph workflow. Triggers the Notice Writer agent to generate official notices.

> **Anti-Conflict Enforced**: A Connections Engineer **cannot** authorize a review that they opened as coordinator.

- **Endpoint**: `POST /reviews/{review_id}/decision`
- **Access**: Connections Engineer only
- **Request Headers**: `X-Session-Token: <token>`, `Content-Type: application/json`
- **Path Parameters**:
  - `review_id` (string, e.g. `REV-LCT0052125-9A4F`)

#### Request Body
```json
{
  "decision": "Approve",
  "decision_reason": "All electrical checks satisfy the Heat Pump Connection Standard with adequate service cut-out headroom."
}
```

#### Response (200 OK)
```json
{
  "review_id": "REV-LCT0052125-9A4F",
  "application_reference": "LCT0052125",
  "technology": "HP",
  "status": "approved",
  "decision": "Approve",
  "decided_by": "amira",
  "decided_by_role": "Connections Engineer",
  "decision_reason": "All electrical checks satisfy the Heat Pump Connection Standard with adequate service cut-out headroom.",
  "installer_notice_text": "MERROWDALE POWER NETWORKS\nCONNECTION APPROVAL NOTICE\n...",
  "household_notice_text": "Dear Customer,\nYour low-carbon heat pump connection has been approved...",
  "llm_call_count": 5
}
```

#### Error Responses
- `404 Not Found`: Review ID does not exist.
- `409 Conflict`: Conflict of interest (decided by opener), decision not among `decisions_open`, or review already decided.

---

### 3.4 Get Review Call Budget
Returns complete telemetry on language model calls made during a review run.

- **Endpoint**: `GET /reviews/{review_id}/calls`
- **Access**: Connections Coordinator, Connections Engineer
- **Request Headers**: `X-Session-Token: <token>`

#### Response (200 OK)
```json
{
  "review_id": "REV-LCT0052125-9A4F",
  "run_id": "RUN-20261112-9A4F",
  "call_count": 5,
  "expected_call_count": 5,
  "total_tokens": 12840,
  "total_latency_milliseconds": 6420,
  "calls": [
    {
      "agent_name": "review_planner_agent",
      "model_name": "gemini-3.5-flash-lite",
      "prompt_tokens": 1420,
      "response_tokens": 310,
      "total_tokens": 1730,
      "latency_milliseconds": 1150
    }
  ]
}
```

---

## 4. Connection Assistant & Audit Trail

### 4.1 Ask Connection Assistant (RAG Chat)
Submits a technical question to the RAG connection assistant. Answers are grounded in the 16 connection standard PDFs, application records, and substation metrics.

- **Endpoint**: `POST /chat/ask`
- **Access**: All Authenticated Roles (Advisors, Coordinators, Engineers)
- **Request Headers**: `X-Session-Token: <token>`, `Content-Type: application/json`

#### Request Body
```json
{
  "question": "What documentation must be provided for a heat pump application?"
}
```

#### Response (200 OK)
```json
{
  "question": "What documentation must be provided for a heat pump application?",
  "question_type": "STANDARD_REQUIREMENT",
  "answer": "Under the Heat Pump Connection Standard, the installer must provide: the heat pump's model and its entry in the approved register, manufacturer data sheet showing rated current with back-up heaters, power factor at full load, single controller description, and maximum demand calculations.",
  "source_documents": [
    "Connection-Standard-Heat-Pump.pdf"
  ],
  "closest_distance": 0.312,
  "was_refused": false
}
```

#### Security Guardrails
If a Customer Service Advisor asks for speculative predictions (e.g. *"Will application LCT0052125 be approved?"*), the assistant refuses the request and responds with policy guidance:
```json
{
  "was_refused": true,
  "answer": "Customer Service Advisors cannot access predictive review assessments. Please check official application status in the records."
}
```

---

### 4.2 Get Own Chat Turns
Retrieves previous questions and answers asked by the signed-in user.

- **Endpoint**: `GET /chat/turns?turn_limit=20`
- **Access**: All Authenticated Roles
- **Request Headers**: `X-Session-Token: <token>`

---

### 4.3 Audit Chat Turns
Retrieves recent questions across all users for auditing purposes.

- **Endpoint**: `GET /audit/chat-turns?turn_limit=50`
- **Access**: Connections Coordinator, Connections Engineer
- **Request Headers**: `X-Session-Token: <token>`
